"""
This file implements get_box_cs and helper functions to prune the bounds of a config space based on metadata.
"""


from pathlib import Path
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter, Integer, Float, InCondition
from typing import Tuple, Union, Dict
import numpy as np
import pandas as pd

from optimization_problem import configuration_space, METADATA_FILE, METADATA_CONFIG_COLUMNS

def get_new_hp_bounds(hp, min_val, max_val, margin):
    """ 
    Helper that computes new boundaries for an HP given the min and max values the new bounds should include,
    including some margin.
    """
    b_low = hp.lower
    b_up = hp.upper

    # we don't prune number of layers
    if hp.name in ["n_fc_layers", "n_conv_layers"]:
        return (b_low, b_up, int(0.5*(b_low + b_up)))

    # for parameters on log-scale we apply the margin on log scale
    if hp.log:
        b_low = np.log(b_low)
        b_up = np.log(b_up)
        min_log = np.log(min_val)
        max_log = np.log(max_val)
        min_log = b_low + (1 - margin) * (min_log - b_low)
        max_log = b_up - (1 - margin) * (b_up - max_log)
        def_log = 0.5 * (min_log + max_log)
        b_low = np.exp(min_log)
        b_up = np.exp(max_log)
        new_default = np.exp(def_log)
    else:
        b_low = b_low + (1 - margin) * (min_val - b_low)
        b_up = b_up - (1 - margin) * (b_up - max_val)
        new_default = 0.5 * (b_low + b_up)
    if type(hp) is UniformFloatHyperparameter:
        return (float(b_low), float(b_up), float(new_default))
    elif type(hp) is UniformIntegerHyperparameter:
        return (int(np.floor(b_low)), int(np.ceil(b_up)), int(new_default))


def adapt_cs_bounds(cs: ConfigurationSpace, new_bounds: Dict):
    """ creates a new configuration space with adapted bounds as provided in the dict"""
    new_cs = ConfigurationSpace()
    
    for hp in cs.get_hyperparameters():
        if type(hp) is UniformIntegerHyperparameter:
            low, up, default = new_bounds[hp.name]
            hp = Integer(hp.name, (low, up), default=default)
        elif type(hp) is UniformFloatHyperparameter:
            low, up, default = new_bounds[hp.name]
            hp = Float(hp.name, (low, up), default=default)
        
        new_cs.add_hyperparameter(hp)
    # as we don't change n_layers we know that we can keep all conditions.
    for cond in cs.get_conditions():
        new_cond = InCondition(new_cs.get_hyperparameter(cond.child.name),
                               new_cs.get_hyperparameter(cond.parent.name),
                               cond.values)
        new_cs.add_condition(new_cond)

    return new_cs


def get_box_cs(cs: ConfigurationSpace, path_to_metadata: Path, margin: float = 0.2,
               outside_the_box: bool = False) -> ConfigurationSpace:
    """Converts the numerical HPs of provided config space to new bounds, such that the boundaries include the
    best configuration per from the provided metadata. If outside the box only upper bounds are being defined. 
    """

    # read the metada and deleted unneeded columns.
    metadata = (
            pd.read_csv(path_to_metadata)
            .astype(METADATA_CONFIG_COLUMNS)
            .rename(columns=lambda c: c.replace("config:", ""))
            .drop(
                columns=[
                    "dataset", "datasetpath", "device", "cv_count", "budget_type", "config_id", "status", "instance",
                    "starttime", "endtime", "time", "scenario_seed"
                ]
            )
        )

    # sort metadata such that we only keep the best configuration per seed.
    metadata = metadata.sort_values(by=["seed", "cost"]).drop_duplicates(subset="seed", keep="first")

    # collect the new lower and upper bounds of the config space in a dictionary.
    low_up_bound_hp = {}

    for hp in cs.get_hyperparameters():
        # only adapt bounds of hps with an order on them (e.g. numerical)
        if type(hp) is UniformFloatHyperparameter or type(hp) is UniformIntegerHyperparameter:
            if outside_the_box:
                # if outside the box only tighten upper bounds on config space.
                if 'n_fc_layers' in hp.name or 'n_conv_layers' in hp.name:
                    min_val = hp.lower
                    max_val = hp.upper
                elif 'learning_rate' in hp.name:
                    min_val = hp.lower
                    max_val = metadata[hp.name].max()
                elif 'n_channels_fc_' in hp.name:
                    min_val = hp.lower
                    max_val = metadata[hp.name].max()
                elif 'n_channels_conv_' in hp.name:
                    min_val = hp.lower
                    max_val = metadata[hp.name].max()
                else:
                    min_val = hp.lower
                    max_val = metadata[hp.name].max()
            else:
                min_val = metadata[hp.name].min()
                max_val = metadata[hp.name].max()
            low_up_bound_hp[hp.name] = get_new_hp_bounds(hp, min_val, max_val, margin)
        else:
            pass

    box_cs = adapt_cs_bounds(cs, low_up_bound_hp)

    return box_cs
