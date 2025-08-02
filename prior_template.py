from __future__ import annotations
from dataclasses import dataclass

import argparse
import json
from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical,
)
from ConfigSpace.api.distributions import Normal
import pandas as pd
import numpy as np
from functools import partial
from typing import Iterator, Literal, Iterable
from pathlib import Path
from dask.distributed import get_worker
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from smac import Scenario, HyperparameterOptimizationFacade
from smac.runhistory import TrialValue
from smac.initial_design import AbstractInitialDesign
from smac.runhistory.dataclasses import TrialInfo
from torch.utils.data import DataLoader, Subset
from datasets import load_deep_woods
import torch
from cnn import Model
from torchvision.datasets import ImageFolder

import logging

from algorithms.fit_distribution import get_weights

logger = logging.getLogger(__name__)
from algorithms.pruned_expected_improvement import EI
from algorithms.local_search_pruned import LocalSearch
from algorithms.prune_searchspace import Prune

from ConfigSpace import (
    CategoricalHyperparameter,
    NormalFloatHyperparameter,
    NormalIntegerHyperparameter

)

# from visualisation.analyse import preprocess_data
HERE = Path(__file__).parent.absolute()
METADATA_FILE = HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"
DATA_PATH = HERE / "datasets"

# These should not be changed when doing the final reporting,
# however feel free to change these while experimenting.
MAX_EPOCHS = 20
IMG_SIZE = 32
CV_SPLITS = 3
DEFAULT_RUNTIME = 60 * 60 * 6
DATASET_NAME = "deepweedsx_balanced"

METADATA_CONFIG_COLUMNS = {
    "config:n_conv_layers": int,
    "config:use_BN": bool,
    "config:global_avg_pooling": bool,
    "config:n_channels_conv_0": int,
    "config:n_channels_conv_1": pd.Int64Dtype(),
    "config:n_channels_conv_2": pd.Int64Dtype(),
    "config:n_fc_layers": int,
    "config:n_channels_fc_0": int,
    "config:n_channels_fc_1": pd.Int64Dtype(),
    "config:n_channels_fc_2": pd.Int64Dtype(),
    "config:batch_size": int,
    "config:learning_rate_init": float,
    "config:kernel_size": int,
    "config:dropout_rate": float,
}


def preprocess_data(metadata_dataframe: pd.DataFrame, columns_to_drop=None, drop_conditional=True):
    # this function drops columns anc preprocesses the data
    #
    if columns_to_drop is None and drop_conditional:
        columns_to_drop = ["status", 'instance', 'budget', 'time', 'status', 'starttime', 'endtime',
                           "n_channels_conv_1", "n_channels_conv_2", "n_channels_fc_1", "n_channels_fc_2"]
    if columns_to_drop is None and not drop_conditional:
        columns_to_drop = ["status", 'instance', 'budget', 'time', 'status', 'starttime', 'endtime', "scenario_seed"]

    metadata_dataframe = metadata_dataframe.drop(
        columns=columns_to_drop)
    print(metadata_dataframe.columns)
    metadata_dataframe["use_BN"] = LabelEncoder().fit_transform(metadata_dataframe["use_BN"])

    # in case of conditional HPs
    # metadata_dataframe = metadata_dataframe.fillna(0)
    return metadata_dataframe


class ProvidedInitialDesign(AbstractInitialDesign):
    """Initial design that uses a user-provided list of configurations."""

    def __init__(self, scenario: Scenario, configs: Iterable[Configuration]):
        self.configs = list(configs)
        super().__init__(scenario=scenario, n_configs=len(self.configs))

    def _select_configurations(self) -> list[Configuration]:
        for config in self.configs:
            config.origin = "Provided Initial Design"

        return self.configs


def get_metadata(
        path: Path,
) -> pd.DataFrame:
    metadata = (
        pd.read_csv(path)
        .astype(METADATA_CONFIG_COLUMNS)
        .rename(columns=lambda c: c.replace("config:", ""))
        .drop(
            columns=[
                "dataset",
                "datasetpath",
                "device",
                "cv_count",
                "budget_type",
                "config_id",
            ]
        )
    )
    return metadata


@dataclass
class WarmstartConfig:
    config: Configuration
    seed: int
    cost: float
    duration: float
    budget: float | None = None

    def as_trial(self) -> tuple[TrialInfo, TrialValue]:
        """Converts this WarmstartConfig into a TrialInfo and TrialValue.

        This can be used with `optimizer.tell(info, value)` to inform SMAC about
        a result before the optimization starts.
        """
        # Since we're not using Multi-fidelity, budget=self.budget,
        trial_info = TrialInfo(config=self.config, instance=None, seed=self.seed)
        trial_value = TrialValue(time=self.duration, cost=self.cost)
        return trial_info, trial_value

    @classmethod
    def from_metadata(
            cls,
            path: Path,
            space: ConfigurationSpace,
    ) -> list[WarmstartConfig]:
        metadata = (
            pd.read_csv(path)
            .astype(METADATA_CONFIG_COLUMNS)
            .rename(columns=lambda c: c.replace("config:", ""))
            .drop(
                columns=[
                    "dataset",
                    "datasetpath",
                    "device",
                    "cv_count",
                    "budget_type",
                    "config_id",
                ]
            )
        )

        config_columns = [c.replace("config:", "") for c in METADATA_CONFIG_COLUMNS]

        configs = []
        for _, row in metadata.iterrows():
            config_dict = row[config_columns].to_dict()  # type: ignore
            try:
                configs.append(
                    WarmstartConfig(
                        config=Configuration(
                            configuration_space=space, values=config_dict
                        ),
                        seed=int(row["seed"]),
                        budget=float(row["budget"]),
                        cost=float(row["cost"]),
                        duration=float(row["time"]),
                    )
                )
            except Exception as e:
                logging.warning(f"Skipping config as not in space:\n{row}\n{e}")

        if len(configs) == 0:
            raise RuntimeError("No configs found that are representable in the space")

        return configs


def get_best_performer_onn_average():
    keys = ["n_channels_conv_0", "n_channels_conv_1", "n_channels_conv_2", "n_fc_layers", "n_conv_layers",
            "n_channels_fc_0", "n_channels_fc_1", "n_channels_fc_2", "batch_size"]

    mu_dict = dict.fromkeys(keys)
    for key in keys:
        grouped_df = metadata_dataframe.groupby(key)['cost'].mean().reset_index()

        # Sort the grouped DataFrame by performance in descending order
        sorted_df = grouped_df.sort_values(by='cost', ascending=True)

        # Select the top performer (top row)
        top_performer = sorted_df.iloc[0]

        # Extract batch size from the top performer's row
        top_val = top_performer[key]
        mu_dict[key] = top_val

    return mu_dict





def weighted_configuration_space() -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    # This serves only as an example of how you can manually define a Configuration Space
    # To illustrate different parameter types;
    # we use continuous, integer and categorical parameters.
    catagorical_hps = []
    sorted_df = metadata_dataframe_.sort_values(by='cost', ascending=True)
    best_configuration = sorted_df.iloc[0]
    mu_dict = get_best_performer_onn_average()

    usebn_weights = get_weights(metadata_dataframe_["use_BN"].sort_values(ascending=False), metadata_dataframe_["cost"])
    pool_weights = get_weights(metadata_dataframe_["global_avg_pooling"].sort_values(ascending=False), metadata_dataframe_["cost"])

    # cs = ConfigurationSpace(
    #     {
    #         "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3,distribution=Normal(mu=3, sigma=0.0001)),
    #         "use_BN": Categorical("use_BN", [True, False], default=True, weights=usebn_weights),
    #         "global_avg_pooling": Categorical(
    #             "global_avg_pooling", [True, False], default=True, weights=pool_weights
    #         ),
    #         "n_channels_conv_0": Integer(
    #             "n_channels_conv_0", (32, 512), default=512, log=True, distribution=Normal(mu=mu_dict["n_channels_conv_0"], sigma=1)
    #         ),
    #         "n_channels_conv_1": Integer(
    #             "n_channels_conv_1", (16, 512), default=512, log=True
    #         ),
    #         "n_channels_conv_2": Integer(
    #             "n_channels_conv_2", (16, 512), default=512, log=True
    #         ),
    #         "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3, distribution=Normal(mu=mu_dict["n_fc_layers"],sigma=1)),
    #         "n_channels_fc_0": Integer(
    #             "n_channels_fc_0", (32, 512), default=512, log=True, distribution=Normal(mu=mu_dict["n_channels_fc_0"], sigma=1)
    #         ),
    #         "n_channels_fc_1": Integer(
    #             "n_channels_fc_1", (16, 512), default=512, log=True
    #         ),
    #         "n_channels_fc_2": Integer(
    #             "n_channels_fc_2", (16, 512), default=512, log=True
    #         ),
    #         "batch_size": Integer("batch_size", (16, 42), default=mu_dict["batch_size"],distribution=Normal(mu=mu_dict["batch_size"], sigma=1), log=True),
    #         "learning_rate_init": Float(
    #             "learning_rate_init",
    #             (1e-5, 0.0074269393683833),
    #             default=1e-3,
    #             log=True,
    #         ),
    #         "kernel_size": Constant("kernel_size", 3),
    #         "dropout_rate": Constant("dropout_rate", 0.2),
    #     }
    # )


    cs = ConfigurationSpace(
        {
            "n_conv_layers": NormalIntegerHyperparameter("n_conv_layers", mu=mu_dict["n_conv_layers"],sigma=0.0001, lower=1, upper=3, default_value=3),
            "use_BN": CategoricalHyperparameter("use_BN", [True, False],weights=usebn_weights, default_value=True),
            "global_avg_pooling": Categorical(
                "global_avg_pooling", [True, False], weights = pool_weights, default=True
            ),
            "n_channels_conv_0": NormalIntegerHyperparameter(
                "n_channels_conv_0", mu=mu_dict["n_channels_conv_0"],sigma=0.0001,lower=32, upper=512, default_value=mu_dict["n_channels_conv_0"], log=True
            ),
            "n_channels_conv_1": Integer(
                "n_channels_conv_1", (16, 512), default=mu_dict["n_channels_conv_1"], log=True
            ),
            "n_channels_conv_2": Integer(
                "n_channels_conv_2", (16, 512), default=mu_dict["n_channels_conv_2"], log=True
            ),
            "n_fc_layers": NormalIntegerHyperparameter("n_fc_layers", mu=mu_dict["n_fc_layers"], sigma=1, upper=3, lower=1, default_value=mu_dict["n_fc_layers"]),
            "n_channels_fc_0": NormalIntegerHyperparameter(
                "n_channels_fc_0", mu=mu_dict["n_channels_fc_0"], sigma=0.0001, lower=32, upper=512, default_value=mu_dict["n_channels_fc_0"], log=True
            ),
            "n_channels_fc_1": Integer(
                "n_channels_fc_1", (16, 512), default=mu_dict["n_channels_fc_1"], log=True
            ),
            "n_channels_fc_2": Integer(
                "n_channels_fc_2", (16, 512), default=mu_dict["n_channels_fc_2"], log=True
            ),
            "batch_size": NormalIntegerHyperparameter("batch_size", mu=mu_dict["batch_size"], sigma=0.0001, lower=1, upper=300, default_value=mu_dict["batch_size"], log=True),
            "learning_rate_init": Float(
                "learning_rate_init",
                (1e-5, 0.0074269393683833),
                default=1e-3,
                log=True,
            ),
            "kernel_size": Constant("kernel_size", 3),
            "dropout_rate": Constant("dropout_rate", 0.2),
        }
    )

    # Add multiple conditions on hyperparameters at once:
    cs.add_conditions(
        [
            InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3]),
            InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3]),
            InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3]),
            InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3]),
        ]
    )
    return cs




@dataclass
class Data:
    """Encapsulates all data functionality

    Notably:
        * `train_val_splits()`: The splits of the training data for cross-validation
        * `train_test()`: The test data and the data to train on when testing
    """

    train_val: ImageFolder
    test: ImageFolder
    cv: StratifiedKFold
    batch_size: int
    random_state: int
    input_shape: tuple[int, int, int]
    classes: list[str]
    folds: int = CV_SPLITS

    @classmethod
    def from_path(
            cls,
            datapath: Path = DATA_PATH,
            batch_size: int = 32,
            download: bool = True,
            img_size: int = IMG_SIZE,
            folds: int = CV_SPLITS,
            seed: int = 0,
    ) -> Data:
        input_shape, train_val, test = load_deep_woods(
            datadir=datapath / "deepweedsx",
            resize=(img_size, img_size),
            balanced=True,
            download=download,
        )
        cv = StratifiedKFold(n_splits=folds, random_state=seed, shuffle=True)
        return Data(
            train_val=train_val,
            test=test,
            cv=cv,
            input_shape=input_shape,
            batch_size=batch_size,
            random_state=seed,
            classes=train_val.classes,
        )

    def train_test(self) -> tuple[DataLoader, DataLoader]:
        train = DataLoader(
            dataset=self.train_val, batch_size=self.batch_size, shuffle=True
        )
        test = DataLoader(dataset=self.test, batch_size=self.batch_size, shuffle=False)
        return train, test

    def train_val_splits(self) -> Iterator[tuple[DataLoader, DataLoader]]:
        splits = self.cv.split(X=self.train_val, y=self.train_val.targets)
        for train_idx, valid_idx in splits:
            train_loader = DataLoader(
                dataset=Subset(self.train_val, list(train_idx)),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                dataset=Subset(self.train_val, list(valid_idx)),
                batch_size=self.batch_size,
                shuffle=False,
            )
            yield train_loader, val_loader


def test_cnn(
        cfg: Configuration,
        seed: int,
        datapath: Path,
        download: bool = True,
        device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    """Function used to get the test score of a CNN.

    Args:
        cfg: Configuration chosen by smac
        seed: used to initialize the rf's random generator
        datapath: path to the dataset

    Returns:
        Test accuracy
    """
    lr = cfg["learning_rate_init"]
    batch_size = cfg["batch_size"]
    model_optimizer = torch.optim.Adam
    train_criterion = torch.nn.CrossEntropyLoss
    # TODO: maybe assert the maxepochs and img size? to ensure theyre not changed when testing

    epochs = MAX_EPOCHS
    img_size = IMG_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    data = Data.from_path(
        datapath=datapath,
        batch_size=batch_size,
        download=download,
        img_size=img_size,
        seed=seed,
    )

    train_set, test_set = data.train_test()
    logging.info(f"Training on {len(train_set)} batches")

    model = Model(
        config=dict(cfg),
        input_shape=data.input_shape,
        num_classes=len(data.classes),
    ).to(model_device)
    optimizer = model_optimizer(model.parameters(), lr=lr)
    criterion = train_criterion().to(device)

    for epoch in range(epochs):
        logging.info("#" * 50)
        logging.info(f"Epoch [{epoch + 1}/{epochs}]")
        train_score, train_loss = model.train_fn(
            optimizer=optimizer,
            criterion=criterion,
            loader=train_set,
            device=model_device,
        )
        logging.info(f"Train acc. {train_score:.3f} | loss {train_loss}")

    test_score = float(model.eval_fn(test_set, device))
    logging.info(f"Test accuracy {test_score:.3f}")

    return test_score


def cnn_from_cfg(
        cfg: Configuration,
        seed: int,
        datapath: Path,
        download: bool = True,
        device: Literal["cpu", "cuda"] = "cpu",
) -> float:
    """Target function optimized to train a CNN on the dataset

    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    Args:
        cfg: Configuration chosen by smac
        seed: used to initialize the rf's random generator
        datapath: path to the dataset

    Returns:
        cross-validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    lr = cfg["learning_rate_init"]
    batch_size = cfg["batch_size"]
    model_optimizer = torch.optim.Adam
    train_criterion = torch.nn.CrossEntropyLoss

    epochs = MAX_EPOCHS
    img_size = IMG_SIZE

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    data = Data.from_path(
        datapath=datapath,
        batch_size=batch_size,
        download=download,
        img_size=img_size,
        seed=seed,
    )
    input_shape = data.input_shape

    score = []

    train_val_splits = data.train_val_splits()
    for cv_index, (train_loader, val_loader) in enumerate(train_val_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")

        model = Model(
            config=dict(cfg),
            input_shape=input_shape,
            num_classes=len(data.classes),
        ).to(model_device)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        criterion = train_criterion().to(device)

        for epoch in range(epochs):
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{epochs}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=criterion,
                loader=train_loader,
                device=model_device,
            )
            logging.info(
                f"Worker:{worker_id} => Train acc. {train_score:.3f} | loss {train_loss}"
            )

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")

        score.append(val_score)

    val_error = float(1 - np.mean(score))  # because minimize
    return float(val_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example using deep weed dataset with naive warmstarting"
    )
    parser.add_argument(
        "--experiment-name",
        default="example",
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
        "--no-test",
        action="store_false",
        help="Whether to evaluate incumbents on the test set after optimization",
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
        default=True,
        help="Whether SMAC should start from scratch and overwrite what's in the experiment dir",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--n-trials", type=int, default=150)
    parser.add_argument("--log-level", default="DEBUG", help="Logging level")
    parser.add_argument("--download", action="store_true", help="Download data")
    parser.add_argument("--metadata-file", type=Path, default=METADATA_FILE)
    parser.add_argument("--evaluate-test", type=bool, default=True)

    args = parser.parse_args()
    metadata_dataframe_ = get_metadata(args.metadata_file)
    metadata_dataframe = preprocess_data(metadata_dataframe_, drop_conditional=False)
    metadata_dataframe = metadata_dataframe.fillna(-1)
    grouped = metadata_dataframe.groupby('seed')
    # convert to
    cost_dataframes = []
    other_dataframes = []

    # Partitioning and processing
    for seed_value, sub_df in grouped:
        # Extracting cost column and other columns
        cost_df = sub_df[['cost']]
        other_df = sub_df.drop(['seed', 'cost'], axis=1)

        # Appending to respective lists
        cost_dataframes.append(cost_df)
        other_dataframes.append(other_df)
    y = [df.to_numpy() for df in cost_dataframes]
    x = [df.to_numpy() for df in other_dataframes]



    logging.basicConfig(level=args.log_level)

    experiment_dir = args.working_dir / args.experiment_name

    logging.info(
        f"Running experiment in {experiment_dir} with the following arguments:\n{args=}"
    )

    configspace = weighted_configuration_space()
    logging.info(f"Using default space\n {configspace}")

    meta_configs = WarmstartConfig.from_metadata(args.metadata_file, space=configspace)
    logger.info(f"Parsed {len(meta_configs)} meta configs that are in the space")

    scenario = Scenario(
        name=args.experiment_name,
        configspace=configspace,
        deterministic=True,
        output_directory=experiment_dir,
        seed=args.seed,
        n_trials=args.n_trials,
        n_workers=args.n_workers,
        walltime_limit=args.runtime,
    )

    target_function = partial(
        cnn_from_cfg,
        seed=args.seed,
        datapath=args.datasetpath,
        device=args.device,
        download=args.download,
    )
    # See: https://github.com/automl/SMAC3/pull/1045
    target_function.__code__ = cnn_from_cfg.__code__  # type: ignore

    from smac.acquisition.function import PriorAcquisitionFunction

    acquisition_function = PriorAcquisitionFunction(
        acquisition_function=HyperparameterOptimizationFacade.get_acquisition_function(scenario),
        decay_beta=scenario.n_trials / 10,  # Proven solid value
    )

    optimizer = HyperparameterOptimizationFacade(
        target_function=target_function,
        scenario=scenario,
        overwrite=args.overwrite,
        logging_level=args.log_level,
        initial_design=None,
        acquisition_function=acquisition_function,
    )

    # Start optimization
    incumbent = optimizer.optimize()
    logging.info("Done!")

    results = {
        "args": args.__dict__.copy(),
        "items": [],
    }

    trajectory = optimizer.intensifier.trajectory
    logging.info(trajectory)

    # Record the trajectory
    for item in trajectory:
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

    results_path = experiment_dir / "results.json"
    logging.info(f"Writing results to {results_path}")
    with results_path.open("w") as fh:
        json.dump(results, fh, indent=4)

    logging.info(f"Finished writing results to {results_path}")

