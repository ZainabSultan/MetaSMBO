"""
Implementing importances_from_lpi with helper functions to determine the most important hyperparameters 
based on DeepCAVEs implementation of LPI.
"""

from deepcave.evaluators.lpi import LPI
from deepcave.runs.converters.smac3v2 import SMAC3v2Run
from deepcave.runs.objective import Objective
from deepcave.evaluators.lpi import LPI
from pathlib import Path
from typing import Tuple, List, Union
from ConfigSpace import ConfigurationSpace
import math
import numpy as np

# local imports
from optimization_problem import WarmstartConfig

import logging

def get_cave_run_from_metadata(path_to_metadata: Path, configspace: ConfigurationSpace) -> SMAC3v2Run:
    """Takes provided Metadata and creates a CAVERun from it to interface to the LPI implementation of CAVE"""
    run = SMAC3v2Run("dummy_run", configspace=configspace, objectives=Objective("cost"))

    # get all the warmstart configs which we can use then use to fill the the run
    warmstart_configs = WarmstartConfig.from_metadata(path_to_metadata, configspace)

    for config in warmstart_configs:
        trial_info, trial_value = config.as_trial()
        config = trial_info.config.get_dictionary()

        run.add(costs=trial_value.cost,
                config=config,
                budget=trial_info.budget)
    
    return run


def importances_from_lpi(cs: ConfigurationSpace, path_to_metadata: Path, importance_threshold=0.8,
                         min_num_hps: int = None, seed: int = 0
                            ) -> Tuple[List[str], List[Tuple[Union[str, Tuple[str, str]], float, float]]]:
    """Determines the hyperparameters to tune for given configspace. Based on LPI importances, computed from the metadata.
    Such that at least importance_threshold is explained and we keep at leat min_num_hps (if None we keep 1/3 of HPs).

    Returns
    -------
    hps_to_tune:
        All hyperparameters that should be tuned accordint to LPI and thresholds.
    importances:
        All computed importances for analytic purposes.
    """    
    # by default we want to keep at least 1/3 of hyperparameters to tune, even if we are above the threshold
    if min_num_hps is None:
        min_num_hps = math.ceil(len(cs.get_hyperparameter_names()) / 3)
    
    # we need the data as cave run to be able use the DeepCAVE LPI implementation
    run = get_cave_run_from_metadata(path_to_metadata, configspace=cs)

    # create LPI and fit forest
    lpi = LPI(run=run)
    lpi.calculate(seed=seed)

    importances_dict = lpi.get_importances(cs.get_hyperparameter_names())
    
    tot_importance = 0
    importances_to_normalize = []

    # extract the mean estimated importances and compute the total sum for normalization later on
    for k, v in importances_dict.items():
        mean = v[0]
        std = np.sqrt(v[1])
        tot_importance += mean
        importances_to_normalize += [(k, mean, std)]
    
    importances_norm = []
    # normalise meand and std of importances
    for imp in importances_to_normalize:
        mean = imp[1] / tot_importance
        std = imp[2] / tot_importance
        importances_norm += [(imp[0], mean, std)]

    # sort importances in descending order
    importances_norm = sorted(importances_norm, key=lambda item: item[1], reverse=True)

    cum_importances = 0
    hps_to_tune = []

    for item in importances_norm:
        if cum_importances < importance_threshold or len(hps_to_tune) < min_num_hps:
            hps_to_tune += [item[0]]
            cum_importances += item[1]
        else:
            break

    logging.info(f"Identified parameters to tune: {hps_to_tune}")
    logging.info(f"Which explain {cum_importances} of local importances.")
    logging.info(f"Overall importances where identified as such: {importances_norm}")

    return hps_to_tune, importances_norm
