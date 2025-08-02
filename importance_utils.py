"""
This file contains the function reduce_config_space_dim and helper functions for it. This function is uses fANOVA or LPI to identify the most important
hyperparameters. The helper functions are used for compute values for parameters that are set to constant (based on the metadata) and 
""" 


from ConfigSpace import ConfigurationSpace, Categorical, Constant, InCondition, UniformIntegerHyperparameter, UniformFloatHyperparameter
from global_importance import importances_from_fanova
from local_importance import importances_from_lpi
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np


def adapt_configspace_to_hp_to_tune(cs: ConfigurationSpace, wishlist_to_tune: List[str], const_vals: Dict) -> ConfigurationSpace:
    """Adapts provided config space such that all its hyperparameters but the ones provided as to tune are set to
    constant. If parents of tunable hps are set to be constant and the const value is not in the condition, parent takes place
    of children. Only supports InConditions and 1-level hierarchies.

    Parameters
    ----------
    cs : ConfigurationSpace
        Config space to transform.
    wishlist_to_tune: List[str]
        Names of hyperparams that should remain tunable.
    const_vals: Dict
        Values to set the frozen hyperparameters to. 

    Returns
    -------
    ConfigurationSpace
        Config space with all hyperparams but those in hp_to_tune set to constant.
    """    

    new_cs = ConfigurationSpace()
    # if children are tunable but parent being const won't satisfy condition, parents take place of children.
    hp_to_tune = []
    cond_to_keep = []
    hp_to_drop = [] # if parent and child are both const and parent will not take value to activate child we won't need the child hyperparam

    # process all unconditional hyperparams from wishlist to hp_to_tune and decide whether they should be tuned
    # by assuming a hierarchy forest of depth 1. Through this we will reach all conditions and hyperparameters
    for hp_name in cs.get_all_unconditional_hyperparameters():
        # if the parent hyperparameter is tunable we can add its children straight away
        if hp_name in wishlist_to_tune:
            hp_to_tune += [hp_name]
            cond_to_keep += cs.get_child_conditions_of(hp_name)
            for cond in cs.get_child_conditions_of(hp_name):
                if cond.child.name in wishlist_to_tune:
                    hp_to_tune += [cond.child.name]
        # else we need to check whether children have even a chance to become active
        else:
            # check if parent will become tunable through any of its children
            becomes_tunable = False
            for cond in cs.get_child_conditions_of(hp_name):
                if const_vals[hp_name] in cond.values:
                    pass
                elif cond.child.name in wishlist_to_tune:
                    becomes_tunable = True

            for cond in cs.get_child_conditions_of(hp_name):
                # the const value of parent satisfies of children to become active
                if const_vals[hp_name] in cond.values:
                    cond_to_keep += [cond]
                    if cond.child.name in wishlist_to_tune:
                        hp_to_tune += [cond.child.name]
                # child is tunable but since it wouldn't become active withouts its parent
                # so parent becomes tunable instead of child.
                elif cond.child.name in wishlist_to_tune:
                    if hp_name not in hp_to_tune:
                        hp_to_tune += [hp_name]
                    cond_to_keep += [cond]
                # it here is no later child that will make the parent tunable
                # child will always inactive and we can remove it
                elif not becomes_tunable:
                    hp_to_drop += [cond.child.name]
                # it will have a chance to become active because another tunable HP makes parent active
                else:
                    cond_to_keep += [cond]

    # process hps depending which list they are in.
    for hp in cs.get_hyperparameters():
        if hp.name in hp_to_drop:
            continue
        elif hp.name in hp_to_tune or type(hp) is Constant:
            new_cs.add_hyperparameter(hp)
        else:
            # apparently ConfigSpace does not support constant boolean hyperparameters so we do a work-around
            if type(hp.default_value) == bool:
                new_cs.add_hyperparameter(Categorical(hp.name, [const_vals[hp.name]], default=const_vals[hp.name]))
            else:
                new_cs.add_hyperparameter(Constant(hp.name, const_vals[hp.name]))
    
    # keep the conditions
    for cond in cond_to_keep:
        new_p = new_cs.get_hyperparameter(cond.parent.name)
        new_c = new_cs.get_hyperparameter(cond.child.name)
        if type(new_p) is Constant:
            new_vals = [new_p.value]
        elif type(new_p) is Categorical and len(new_p.choices) == 1:
            new_vals = [new_p.default_value]
        else:
            new_vals = cond.values
        new_condition = InCondition(child=new_c, parent=new_p, values=new_vals)
        new_cs.add_condition(new_condition)
    
    return new_cs


def get_const_value_candidates(cs: ConfigurationSpace, path_to_metadata: Path) -> Dict:
    """When setting the values of some hyperparameters to constant we pick the best value in the metadata-set
    after averaging over all other hyperparameters to set to. 
    """
    metadata = pd.read_csv(path_to_metadata)
    metadata.columns = [col.replace("config:", "") for col in metadata.columns]

    const_values = {}

    for hp in cs.get_hyperparameters():
        grouped_df = metadata.groupby(hp.name)['cost'].mean().reset_index()

        sorted_df = grouped_df.sort_values(by='cost', ascending=True)

        # Select the top performer (top row)
        top_performer = sorted_df.iloc[0]

        # Extract batch size from the top performer's row
        top_val = top_performer[hp.name]

        # need to make sure to convert to the right datatype
        # need this because we want real bool and not numpy.bool_ (will break json otherwise)
        if type(top_val) == np.bool_:
            top_val = bool(top_val)
        elif type(hp) == UniformIntegerHyperparameter:
            top_val = int(top_val)
        elif type(hp) == UniformFloatHyperparameter:
            top_val = float(top_val)
        
        const_values[hp.name] = top_val

    return const_values


def reduce_config_space_dim(cs: ConfigurationSpace, path_to_metadata: Path, method: str='lpi',
                            impute_cond=True, impute_strat: str='median', importance_threshold=0.8,
                            seed: int=0,
                            uncond_only: bool = False) -> Tuple[ConfigurationSpace, List[Tuple[str, float, float]]]:
    """
    Reduces number of tunable hyperparameters in the config space based on hyperparameter importances from metadata
    under provided path. Will set all hyperparameters but the most important ones to constant. And return the config
    space that was transformed accordingly. Different methods to compute the importances are available. Hyperparameter
    importances are considered up to 2nd order interactions.

    Parameters:
    -----------
    cs: ConfigurationSpace
        ConfigurationSpace to convert.
    path_to_csv: Path
        Path where metadata collected on provided config space is stored.
    method: str
        Method to use to compute the hyperparameter importances. Choose from {'fanova' 'shapley'}
    impute_cond: bool
        Whether to impute missing values of conditional hyperparameters. HPs with an order on them are imputed using the
        median. If we do not impute we have to remove all conditional hyperparameters from fanova analysis and set them
        constant straight away.
    impute_strat: str \in {'smac', 'median', 'default'}
        How to impute values of inactive hyperparams. 'smac' replaces numerical hyperparams (float, int) with -1, and for
        categorical introduces a new index. 'median' sets numerical value to the median of values for this feature.
        'default' sets it to the HPs specified default value.
    seed: int
        Used for seeding the importance computations in fANOVA or LPI which rely on random forests.

    Returns:
    --------
    red_cs: ConfigurationSpace
        Config space as cs, but all hyperparameters but the most important ones are set to constant for their default
        values as defined in cs.
    importances: List
        List of tuples ((hp_names,), individual_importance, individual_std) which can be used to analyse the computed
        importances, e.g. for plotting.
    """
    if method == 'fanova':
        hps_to_tune, importances = importances_from_fanova(
            cs, path_to_metadata, importance_threshold=importance_threshold, impute_cond=impute_cond, seed=seed,
            impute_strat=impute_strat, uncond_only=uncond_only)
    elif method == "lpi":
        hps_to_tune, importances = importances_from_lpi(cs, path_to_metadata, importance_threshold=importance_threshold,
                                                        seed=seed)
       
    # now it remains to convert the config space into reduced config space by fixing all hyperparameters to const that
    # have not been selected for tuning.
    const_values = get_const_value_candidates(cs, path_to_metadata)
    new_cs = adapt_configspace_to_hp_to_tune(cs, wishlist_to_tune=hps_to_tune, const_vals=const_values)

    return new_cs, importances


if __name__ == "__main__":
    from optimization_problem import METADATA_FILE, configuration_space, WarmstartConfig
    import numpy as np

    import logging

    logging.basicConfig()
    cs = configuration_space()
    warmstart_cofigs = WarmstartConfig.from_metadata(METADATA_FILE, cs)
    cs, importances = reduce_config_space_dim(cs, path_to_metadata=METADATA_FILE, method='lpi', seed=0)
    print(importances)
