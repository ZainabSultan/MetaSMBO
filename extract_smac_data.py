from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import os
import logging
import json

TIME_ARRAY = np.linspace(0, 21600, 217)

def smac_val_test_results_to_np(exp_name: str, timescale: np.ndarray, first_inc_idx: int = 0):
    """
    Function that extracts the results from all seeds of an experiment and returns incument validation and test accuracy trajectory.
    """
    path_to_experiment = Path(__file__).resolve().parent / 'smac3_output' / exp_name

    seeds = []
    for seed in os.listdir(path_to_experiment):
        try:
            seeds += [int(seed)]
        except:
            logging.info(f"Passing subdirectory {seed} of experiment {exp_name} as it seems not to be a seeded run.")
    
    incumbent_test = np.ndarray((len(seeds), len(timescale)))
    incumbent_val = np.ndarray(incumbent_test.shape)

    for i, seed in enumerate(seeds):
        path_to_results = path_to_experiment / str(seed) / 'results.json'

        with open(path_to_results, 'r') as file:
            results = json.load(file)


        current_test_acc = np.nan
        current_val_acc = np.nan
        start_idx = 0
        for res in results['items'][first_inc_idx:]:
            if res['config_id'] > 150:
                # some runs where assigned more than 150 trials but for fair comparison we don't
                # include them
                logging
                continue
            walltime = res['walltime']
            end_idx = np.argmax(TIME_ARRAY >= walltime)
            incumbent_test[i][start_idx:end_idx] = current_test_acc
            incumbent_val[i][start_idx:end_idx] = current_val_acc

            start_idx = end_idx
            current_val_acc = res['val-acc']
            current_test_acc = res['test-acc']
        incumbent_val[i][start_idx:] = current_val_acc
        incumbent_test[i][start_idx:] = current_test_acc

    return incumbent_val, incumbent_test
