"""
File that implements importances_from_fanova with helper functions. It interfaces fANOVA to determine the most important hyperparameters
that should be kept tunable in the hyperparameter space.
"""

from fanova import fANOVA
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Union
from ConfigSpace import ConfigurationSpace, Constant, Integer, UniformIntegerHyperparameter
from itertools import combinations
import logging
import math

logging.getLogger(__name__)


path_to_metadata = Path(".").absolute() / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"


def remove_conditional(cs: ConfigurationSpace) -> ConfigurationSpace:
    new_cs = ConfigurationSpace()
    conditional_hps = cs.get_all_conditional_hyperparameters()

    for hp in cs.get_hyperparameters():
        if hp.name in conditional_hps:
            continue
        else:
            new_cs.add_hyperparameter(hp)
    return new_cs


def remove_constant(cs: ConfigurationSpace) -> ConfigurationSpace:
    new_cs = ConfigurationSpace()
    for hp in cs.get_hyperparameters():
        if type(hp) == Constant:
            continue
        else:
            new_cs.add_hyperparameter(hp)
    new_cs.add_conditions(cs.get_conditions())
    return new_cs


def extend_conditional_range(cs: ConfigurationSpace) -> ConfigurationSpace:
    hps_to_extend = []
    for cond in cs.get_conditions():
        hps_to_extend += [cond.child]

    new_cs = ConfigurationSpace()
    for hp in cs.get_hyperparameters():
        # only assuming to encounter hyperparameters of type Integer here (channels)
        if hp in hps_to_extend:
            if type(hp) is not UniformIntegerHyperparameter:
                raise NotImplementedError(f"As of now only Integer hyperparameters can be extended."
                                          + f"Provided HP is of type {type(hp)}")
            hp_add = Integer(hp.name, (-1, hp.upper), default=hp.default_value)
            new_cs.add_hyperparameter(hp_add)
        else:
            new_cs.add_hyperparameter(hp)
    
    return new_cs


def format_metadata_for_fanova(path_to_csv: Path, cs: ConfigurationSpace, impute_cond: bool=True, 
                               impute_strat: str='median') -> Tuple[np.ndarray, np.ndarray, ConfigurationSpace]:
    """
    Processes metadata from deepweedsx such that we have all relevant information from the metadata in csv to run fanova.

    Parameters:
    -----------
    path_to_csv: Path
        Path where metadata is stored

    Returns:
    --------
    X: np.ndarray
        All configs stored in row-major style, e.g. shape is (n_configs, n_hyperparams)
    Y: np.ndarray
        Cost averaged per config over seeds, shape is (n_configs, 1)
    hp_to_idx: Dict
        Maps the hyperparameter name to the corresponding column idx of X.
    impute_strat: str \in {'smac', 'median', 'default'}
        How to impute values of inactive hyperparams. 'smac' replaces numerical hyperparams (float, int) with -1, and for
        categorical introduces a new index. 'median' sets numerical value to the median of values for this feature.
        'default' sets it to the HPs specified default value.
    impute_cond: bool
        If we don't impute conditional hyperparameter values we need to remove them because fANOVA won't accept inactive hyper
        parameters.
    """
    # remove constant hyperparameters
    cs = remove_constant(cs)
    metadata = pd.read_csv(path_to_csv)
    metadata.columns = [col.replace("config:", "") for col in metadata.columns]

    if impute_cond:
        cols_impute = metadata.filter(regex="channels", axis=1)
        if impute_strat == 'smac':
            impute_vals = -1

            # in this case we also have to extend the range of the hyperparameters, because there lower bound is
            # above -1.
            cs = extend_conditional_range(cs)
        elif impute_strat == 'median':
            # assuming linear dependence between n_channels and performance we take the median of all existing conditional values to impute NaNs
            # we will be able to impute only channels
            impute_vals = cols_impute.median()
        elif impute_strat == 'default':
            impute_vals = {}
            for hp in cs.get_hyperparameters():
                if "channel" in hp.name:
                    impute_vals[hp.name] = hp.default_value
        else:
            raise ValueError(f"Cannot use {impute_strat} to impute values of inactive hyperparameters.")
        metadata[cols_impute.columns] = cols_impute.fillna(impute_vals)
    else:
        # if values of inactive hyperparameters are not imputed we have to delete them, else fANOVA won't except.
        cs = remove_conditional(cs)
        
    config_columns = cs.get_hyperparameter_names()
    metadata = metadata.filter(config_columns + ["cost"])
    metadata = metadata.groupby(config_columns, dropna=False).mean().reset_index().sort_values(config_columns)

    # configs are sorted and we can extract the costs as target now
    Y = metadata.drop(config_columns, axis=1).to_numpy()

    # to get the relevant config features as numpy array while storing column indices we need to perform some more
    # steps
    X = metadata.drop(["cost"], axis=1).to_numpy().astype('float64')

    return X, Y, cs


def importances_from_fanova(cs: ConfigurationSpace, path_to_metadata: Path, impute_cond=True,
                            impute_strat: str='median', importance_threshold=0.7, min_num_hps: int = None, seed: int=0,
                            uncond_only: bool = False
                            ) -> Tuple[List[str], List[Tuple[Union[str, Tuple[str, str]], float, float]]]:
    """Function that returns the most important hyperparameters that should be kept tunable, with importances being computed
    through fANOVA.

    Parameters
    ----------
    cs : ConfigurationSpace
        Config Space to compute the importances for.
    path_to_metadata : Path
    impute_cond : bool, optional
        If we don't impute conditional hyperparameter values we need to remove them because fANOVA won't accept inactive hyper
        parameters.
    impute_strat : str, optional
        How to impute values of inactive hyperparams. 'smac' replaces numerical hyperparams (float, int) with -1, and for
        categorical introduces a new index. 'median' sets numerical value to the median of values for this feature.
        'default' sets it to the HPs specified default value.
    importance_threshold : float, optional
        Importance that has ti be explained be explained by , by default 0.7
    min_num_hps: int = None
        Min num of hyperparameters to keep tunable regardless of importance. If None keeps 1/3 of HPs.

    Returns
    -------
    hps_to_tune: List[str]
        List of hyperparameters most important hyperparameters explaining at least importance_threshold importance.
    importances: List[Tuple[str, float, float]]]
        List with importances computed for all hyperparameters
    """    
    # by default we want to keep at least 1/3 of hyperparameters to tune, even if we are above the threshold
    if min_num_hps is None:
        min_num_hps = math.ceil(len(cs.get_hyperparameter_names()) / 3)
    
    # we might need a cs where all conditional hyperparameters have been removed (if we do not impute)
    X, Y, cs_for_fanova = format_metadata_for_fanova(path_to_metadata, cs, impute_cond=impute_cond, impute_strat=impute_strat)
    f = fANOVA(X, Y, cs_for_fanova, seed=seed)

    # compute and store hyperparameter importances up to 2nd order interaction in a list
    importances = []
    if uncond_only:
        hps_to_evaluate = cs_for_fanova.get_all_unconditional_hyperparameters()
    else:
        hps_to_evaluate = cs_for_fanova.get_hyperparameter_names()
    hps = list(combinations(hps_to_evaluate, 1))  # uniform dict access
    hp_pairs = list(combinations(hps_to_evaluate, 2))
    
    # we compute the importances relative to the total importance explained by 2nd order interaction.
    achievable_importance = 0
    for hp in hps + hp_pairs:
        res = f.quantify_importance(hp)[hp]
        importances += [(hp, res['individual importance'], res['individual std'])]
        achievable_importance += res['individual importance']

    # order the hyperparams in descending order by their importance
    importances = sorted(importances, key=lambda it: it[1], reverse=True)
    
    # identify the parameters that, including 2nd order interaction effects explain importance above threshold
    explained_importance = 0
    checksum = 0
    hps_to_tune = []
    normalized_importances = []
    for tup in importances:
        # while iterating normalize mean and std with achievable importance
        norm_mean = tup[1] / achievable_importance
        norm_std = tup[2] /achievable_importance
        
        if explained_importance < importance_threshold or len(hps_to_tune) < min_num_hps:
            explained_importance += norm_mean
            for hp in tup[0]:
                if hp not in hps_to_tune:
                    hps_to_tune.append(hp)
        normalized_importances += [(tup[0], norm_mean, norm_std)]
        checksum += norm_mean

    print(f"Importances summing up to {checksum}")
    logging.warning(f"Parameters of interest up to 2nd order interactions couldn't explain importance above threshold {importance_threshold}")
    logging.warning(f"They only explain importance of {explained_importance}")
    logging.info(f"Identified parameters to tune: {hps_to_tune}")
    logging.info(f"Which explain {explained_importance} of global importance and are considering up to 2nd order interaction.")
    logging.info(f"Overall importances where identified as such: {normalized_importances}")

    return hps_to_tune, normalized_importances
