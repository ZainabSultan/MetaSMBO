from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from smac import Scenario, HyperparameterOptimizationFacade
from smac.initial_design import DefaultInitialDesign, SobolInitialDesign
from smac.intensifier.intensifier import Intensifier
import torch
from utils import serialize_path_json
from optimization_problem import (
    DEFAULT_RUNTIME,
    METADATA_FILE,
    configuration_space,
    WarmstartConfig,
    cnn_from_cfg,
    test_cnn
)
from importance_utils import reduce_config_space_dim
from box_pruning import get_box_cs

import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example using deep weed dataset with naive warmstarting"
    )
    parser.add_argument(
        "--experiment-name",
        default="spacepruning",
        type=str,
        help="The unique name of the experiment",
    )
    parser.add_argument(
        "--working-dir",
        default=Path(".").absolute(),
        type=Path,
        help="The base path SMAC will run from",
    )
    parser.add_argument(
        "--runtime",
        default=DEFAULT_RUNTIME,
        type=int,
        help="Max running time (seconds) allocated to run HPO",
    )
    parser.add_argument(
        "--datasetpath",
        type=Path,
        default="data",
        help="Path to directory containing the dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether SMAC should start from scratch and overwrite what's in the experiment dir",
    )
    parser.add_argument(
        "--warmstart-smbo",
        action="store_true",
        help="Runs the greedy warmstart.",
    )
    parser.add_argument(
        "--warmstart-all-in",
        action="store_true",
        help="Uses all metadata to fit the model but still samples 37 Sobol initial configs.",
    )
    parser.add_argument(
        "--box-pruning",
        action="store_true",
        help="Uses box_pruning to limit the ranges of all numerical hyperparameters.",
    )
    parser.add_argument(
        "--box-outside",
        action="store_true",
        help="Only adapts the upper bounds of config space according to the metadata (alternative box).",
    )
    parser.add_argument(
        "--fanova-reduce-dim",
        action="store_true",
        help="Whether HPO should be restricted to the most important hyperparameters, based on fANOVA analysis of meta-data.",
    )
    parser.add_argument(
        "--lpi-reduce-dim",
        action="store_true",
        help="Whether HPO should be restricted to the most important hyperparameters, based on LPI analysis of meta-data.",
    )
    parser.add_argument(
        "--evaluate-test",
        default=True,
        type=bool,
        help="Whether to evaluate all incumbent configurations on the test dataset."
    )
    parser.add_argument("--max-config-calls",
                        default=1,
                        type=int,
                        help="No of seeds to use by racing.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--log-level", default='INFO', help="Logging level in \{'DEBUG', 'INFO'\}")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)

    args = parser.parse_args()

    # it's a bit a hack to properly parse the log level from the CLI so we compare the string here.
    if args.log_level == "INFO":
        log_level = logging.INFO
    elif args.log_level == "DEBUG":
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(level=log_level)


    experiment_dir = args.working_dir / "smac3_output"

    logging.info(
        f"Running experiment in {experiment_dir} with the following arguments:\n{args=}"
    )

    if args.device == "cuda":
        logging.info(f"Using the GPU and it is available: {torch.cuda.is_available()}")
        logging.info(f"Number of available GPUs: {torch.cuda.device_count()}")
    else:
        logging.info(f"Running without GPU.")

    # switches to prune the config space.
    configspace = configuration_space()
    if args.box_pruning:
        configspace = get_box_cs(configspace, METADATA_FILE, margin=0.2)
        logging.info("#" * 50)
        logging.info("Pruned the configspace numerical hyperparameters to include best metadata:")
        logging.info(configspace)
    elif args.box_outside:
        configspace = get_box_cs(configspace, METADATA_FILE, margin=0.2, outside_the_box=True)
        logging.info("#" * 50)
        logging.info("Pruned the configspace numerical hyperparameters to exclude best metadata:")
        logging.info(configspace)

    # switches to reduve the dimensions of the configspace.
    logging.info("#" * 50)
    if args.lpi_reduce_dim:
        configspace, _ = reduce_config_space_dim(configspace, path_to_metadata=args.metadata_file, method='lpi', seed=args.seed)
        logging.info(f"Using config space with reduced dimensions:\n {configspace}. " 
                     + f"Obtained by using importances from LPI.")
    elif args.fanova_reduce_dim:
        configspace, _ = reduce_config_space_dim(configspace, path_to_metadata=args.metadata_file, method='fanova', seed=args.seed)
        logging.info(f"Using config space with reduced dimensions:\n {configspace}. " 
                     + f"Obtained by using importances from fanova.")
    else:
        logging.info(f"Keeping all tunable dimensions of the configuration space:\n {configspace}")
    logging.info("#" * 50)

    # if we choose warmstarting we can get the configs for the tell iterface here.
    if args.warmstart_smbo:
        tell_meta_configs = WarmstartConfig.from_metadata(args.metadata_file, space=configspace,
                                                          limit_seeds=args.max_config_calls)
    elif args.warmstart_all_in:
        tell_meta_configs = WarmstartConfig.from_metadata(args.metadata_file, space=configspace, hallucinate=True)
    else:
        tell_meta_configs = []
    logging.info(f"Parsed {len(tell_meta_configs)} meta configs to tell the optimizer that are in the space")

    scenario = Scenario(
        name=args.experiment_name,
        configspace=configspace,
        deterministic=(args.max_config_calls == 1),
        output_directory=experiment_dir,
        seed=args.seed,
        n_trials=args.n_trials + len(tell_meta_configs),  # meta-configs provided "free" -> don't count into n_trials
        n_workers=args.n_workers,
        walltime_limit=args.runtime,
    )

    # specify intensifier and max_config_calls explicitly to avoid configs reentering queue for racing with higher N
    intensifier = Intensifier(scenario=scenario, max_config_calls=args.max_config_calls)

    
    if args.warmstart_smbo:
        # if pure warmstart we want to sample from surrogate right away. As SMAC won't accept 0 initial configurations
        # default initial design is the cheapest option with only 1 evaluation involved.
        initial_design = DefaultInitialDesign(scenario=scenario)
    else:
        # we might specify some initial configurations in the future so to be save, lets fix the number of initial_configurations
        initial_design = SobolInitialDesign(scenario=scenario, n_configs=37)

    target_function = partial(
        cnn_from_cfg,
        seed=args.seed,
        datapath=args.datasetpath,
        device=args.device,
        download=args.download,
    )
    # See: https://github.com/automl/SMAC3/pull/1045
    target_function.__code__ = cnn_from_cfg.__code__  # type: ignore

    optimizer = HyperparameterOptimizationFacade(
        target_function=target_function,
        scenario=scenario,
        overwrite=args.overwrite,
        logging_level=log_level,
        initial_design=initial_design,
        intensifier=intensifier,
    )

    # if warmstarting tell the warmstart configs now.
    for config in tell_meta_configs:
        optimizer.tell(*config.as_trial())
    
    logging.info("####### Starting SMBO. #######")
    # Start optimization
    incumbent = optimizer.optimize()
    logging.info("####### Finished SMBO. #######")

    results = {
        "args": args.__dict__.copy(),
        "items": [],
    }

    trajectory = optimizer.intensifier.trajectory
    logging.info(trajectory)

    # Record the trajectory, if we used warmstarting only starting with the last
    # incumbent from the warmstart dataset, else it might take too long.
    for item in trajectory[len(tell_meta_configs)-1:]:
        config_id = item.config_ids[0]
        config = optimizer.runhistory.get_config(config_id)

        val_cost = item.costs[0]
        assert not isinstance(val_cost, list)

        entry = {
            "val-acc": float(1 - val_cost),
            "walltime": item.walltime,
            "config_id": config_id,
            "config": dict(config),
        }

        if args.evaluate_test:
            try:
                test_accuracy = test_cnn(
                    config,
                    seed=args.seed,
                    datapath=args.datasetpath,
                    device=args.device,
                    download=args.download,
                )
            except Exception as e:
                logging.exception(e)
                test_accuracy = 0.0

            entry["test-acc"] = test_accuracy

        results["items"].append(entry)

    results_path = scenario.output_directory / "results.json"
    logging.info(f"Writing results to {results_path}")
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=4, default=serialize_path_json)

    logging.info(f"Finished writing results to {results_path}")
