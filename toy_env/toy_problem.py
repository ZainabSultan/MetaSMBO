import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float, NormalIntegerHyperparameter, Integer
from matplotlib import pyplot as plt
from smac.model.random_forest.random_forest import RandomForest
from smac.initial_design.sobol_design import SobolInitialDesign
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario
from smac.initial_design import AbstractInitialDesign, DefaultInitialDesign
from algorithms.local_search_pruned import LocalSearch
from visualisation.analyse import preprocess_data
from algorithms.prune_searchspace import Prune
from ConfigSpace.api.distributions import Normal
import argparse
from smac.acquisition.function import PriorAcquisitionFunction
from prune_space_template import get_metadata, HERE
from pathlib import Path
from algorithms.pruned_expected_improvement import EI
METADATA_FILE = HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"
DATA_PATH = HERE / "datasets"


metadata_dataframe_ = get_metadata(METADATA_FILE)
metadata_dataframe = preprocess_data(metadata_dataframe_, drop_conditional=False)
metadata_dataframe=metadata_dataframe.fillna(-1)



def get_best_performer_onn_average(metadata_dataframe_):
    keys = ["n_channels_conv_0", "n_channels_conv_1", "n_channels_conv_2", "n_fc_layers", "n_conv_layers",
            "n_channels_fc_0", "n_channels_fc_1", "n_channels_fc_2", "batch_size"]

    mu_dict = dict.fromkeys(keys)
    for key in keys:
        grouped_df = metadata_dataframe_.groupby(key)['cost'].mean().reset_index()

        # Sort the grouped DataFrame by performance in descending order
        sorted_df = grouped_df.sort_values(by='cost', ascending=True)

        # Select the top performer (top row)
        top_performer = sorted_df.iloc[0]

        # Extract batch size from the top performer's row
        top_val = top_performer[key]
        mu_dict[key] = top_val

    return mu_dict

mu_dict = get_best_performer_onn_average(metadata_dataframe_)
print(mu_dict)
class Rosenbrock2D:
    @property
    def old_configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x0 = Float("x0", (-5, 10), default=-3)
        x1 = Float("x1", (-5, 10), default=-4)
        cs.add_hyperparameters([x0, x1])

        return cs

    @property
    def configspace(self) -> ConfigurationSpace:

        dist_1 = Normal(mu=mu_dict["n_conv_layers"],sigma=0.001
                                                     )
        dist_2= Normal(
         mu=mu_dict["n_channels_conv_0"], sigma=1,

        )

        cs =ConfigurationSpace({

            "x1": Integer("x1", (16, 126), default=mu_dict["batch_size"],
                                  distribution=Normal(mu=mu_dict["batch_size"], sigma=1)),

            # "x1":NormalIntegerHyperparameter("n_conv_layers", mu=mu_dict["n_conv_layers"],sigma=0.001, lower=1, upper=3, default_value=3),

        "x0": Integer(
                "x0", (-5, 10), default=0
            ),
        # "n_fc_layers": NormalIntegerHyperparameter("n_fc_layers", mu=mu_dict["n_fc_layers"], sigma=1, upper=3, lower=1,
        #                                            default_value=mu_dict["n_fc_layers"]),
        # "n_channels_fc_0": NormalIntegerHyperparameter(
        #     "n_channels_fc_0", mu=mu_dict["n_channels_fc_0"], sigma=1, lower=32, upper=512,
        #     default_value=mu_dict["n_channels_fc_0"], log=True
        # ),
        #
        # "batch_size": NormalIntegerHyperparameter("batch_size", mu=mu_dict["batch_size"], sigma=1, lower=1, upper=300,
        #                                           default_value=mu_dict["batch_size"], log=True)
        })

        return cs


    def train(self, config: Configuration, seed: int = 0) -> float:
        """The 2-dimensional Rosenbrock function as a toy model.
        The Rosenbrock function is well know in the optimization community and
        often serves as a toy problem. It can be defined for arbitrary
        dimensions. The minimium is always at x_i = 1 with a function value of
        zero. All input parameters are continuous. The search domain for
        all x's is the interval [-5, 10].
        """
        x1 = config["x0"]
        x2 = config["x1"]
        print(x2, "this is x2")

        if x2 is not None:
            assert x2 <= 126

        cost = 10.0 * (x2 - x1**2.0) ** 2.0 + (1 - x1) ** 2.0
        if cost is None:
            print("wrong returns")
        return cost
class ToyProblem:
    def __init__(self, weights, noise_params, type="paraboliod"):
        self.weights = weights
        self.noise_params = noise_params
        self.type = type

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        x1 = Float("x1", (-10, 10), default=-10)
        x2 = Float("x2", (-10, 10), default=-10)
        cs.add_hyperparameters([x1, x2])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        """Returns the y value of a paraboliod x1^2 + x2^2 function with a minimum we know to be at x=0."""
        x1 = config["x1"]
        x2 = config["x2"]
        # sample gaussian noise centered around noise_params[0] and variance = noise_params[1]
        mu = self.noise_params[0]
        sigma = self.noise_params[1]
        noise = np.random.normal(mu, sigma, 1)[0]

        return noise + (self.weights[0] * (x1 ** 2) + self.weights[1] * (x2 ** 2))

    def func_definition(self, X1, X2):
        return self.weights[0] * (X1 ** 2) + self.weights[1] * (X2 ** 2)


class InitialDesign(AbstractInitialDesign):
    def __init__(self, scenario: Scenario):
        super().__init__(scenario)


def plot_contour(runhistory: RunHistory, intensifier, incumbent: Configuration, model, save_dir) -> None:
    weights = model.weights
    # Plot ground truth
    x = np.linspace(-10, 10, 100)

    y = np.array([xi ** 2 + xi ** 2 for xi in x])

    X1, X2 = np.meshgrid(x, x)
    Y = model.func_definition(X1, X2)
    fig, ax = plt.subplots()
    CS = ax.contour(X1, X2, Y)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Simplest default with labels')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('surface')

    incumbents_list = []
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x1 = config["x1"]
        x2 = config["x2"]
        y = v.cost  # type: ignore
        # incumbents_list.append((x1, x2, y, time))
        ax.scatter(x1, x2, c="blue", alpha=0.1, marker="o")

    # Plot incumbent
    # plot trajectory
    traj_idx = 0

    for trajectory_item in intensifier.trajectory:
        for config_id in trajectory_item.config_ids:
            config = runhistory.get_config(config_id)
            x1 = config["x1"]
            x2 = config["x2"]
            incumbents_list.append((x1, x2, traj_idx))
        traj_idx += 1
    mappable = ax.scatter([x1 for x1, _, _ in incumbents_list], [x2 for _, x2, _ in incumbents_list],
                          c=[t for _, _, t in incumbents_list], cmap='Greens')
    ax.scatter(incumbent["x1"], incumbent["x2"], c="red", marker="x")

    fig.colorbar(mappable, label="time", orientation="horizontal")
    plt.savefig(
        save_dir + "trials_{weights}_PCA{pca_comps}_noise_{noise_params}_nc{n_configs}.png".format(weights=str(weights),
                                                                                                   pca_comps=pca_components
                                                                                                   ,
                                                                                                   n_configs=n_configs,
                                                                                                   noise_params=str(
                                                                                                       model.noise_params[
                                                                                                           0]) + str(
                                                                                                       model.noise_params[
                                                                                                           1])))
    plt.show()


def plot(runhistory: RunHistory, incumbent: Configuration) -> None:
    plt.figure()

    # Plot ground truth
    x = np.linspace(-5, 5, 100)

    y = np.array([xi ** 2 + xi ** 2 for xi in x])

    X1, X2 = np.meshgrid(x, x)
    Y = X1 ** 2 + X2 ** 2
    ax = plt.axes(projection='3d')
    # ax.plot_surface(X1, X2, Y, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.contour3D(X1, X2, Y, 50, cmap='binary')
    ax.scatter(x, x, y, c=y, cmap='viridis', linewidth=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('surface')

    # Plot all trials
    for k, v in runhistory.items():
        config = runhistory.get_config(k.config_id)
        x1 = config["x1"]
        x2 = config["x2"]
        y = v.cost  # type: ignore
        ax.scatter3D(x1, x2, y, c=y, cmap='Greens')

    # Plot incumbent
    print(incumbent["x1"], incumbent["x2"])
    ax.scatter3D(incumbent["x1"], incumbent["x2"], incumbent["x1"] ** 2 + incumbent["x2"] ** 2, c="red", marker="x")

    # plt.scatter(incumbent["x1"], incumbent["x"] * incumbent["x"], c="red", zorder=10000, marker="x")

    plt.show()


def plot_progress(smac_run: HPOFacade):
    traj = smac_run.intensifier.trajectory
    num_trials = smac_run.scenario.n_trials

    print(traj)
    performance = np.full(num_trials, np.nan)

    for curr_incumbent in traj:
        conf_id = curr_incumbent.config_ids[0]
        performance[conf_id] = curr_incumbent.costs[0]

    for i in range(1, len(performance)):
        if np.isnan(performance[i]):
            performance[i] = performance[i - 1]
    plt.plot(performance)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Example using deep weed dataset with naive warmstarting"
    )
    parser.add_argument(
        "--datasetpath",
        type=Path,
        default="data",
        help="Path to directory containing the dataset",
    )
    parser.add_argument("--metadata_file", type=Path, default=METADATA_FILE)
    args = parser.parse_args()


    # creating datasets by partitioning on seeds
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
    #datasets = [sub_df.drop('seed', axis=1).values for _, sub_df in grouped]


    #y = metadata_dataframe["cost"]
    #x = metadata_dataframe.drop(columns=["cost"])  # input features
    # Grouping the DataFrame by the "seed" column

    # Converting each sub DataFrame to a NumPy array after dropping the "seed" column
    x=[[[0.8976636599379368,0.5778983950580887],[-4.662429578602314, 2.4112939089536667],[2.6820622757077217,-5.740029886364937],[2.055267521432878,5.834500761653292]]]
    y= [[4000, 1307, 434, 238]]

    #pruner = Prune(x, y)
    #acq = EI(pruner=pruner)

    pca_components = 2
    n_configs = 1
    #model = ToyProblem(weights=[60, 0], noise_params=(0.0, 10))  # 4.371640922850756e-08
    model = Rosenbrock2D()

    n_trials=150


    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, deterministic=True, n_trials=n_trials, n_workers=2)

    # change PCA components
    #surrogate_model = RandomForest(n_trees=10, pca_components=pca_components, configspace=model.configspace)


    # change initial design number of HPs
    initial_design = SobolInitialDesign(scenario=scenario)
    # Now we use SMAC to find the best hyperparameters
    smac = HPOFacade(
        scenario,
        model.train,  # We pass the target function here
        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
        #model=surrogate_model,


        #initial_design=initial_design,
        acquisition_function=PriorAcquisitionFunction(decay_beta=n_trials/10, acquisition_function=HPOFacade.get_acquisition_function(scenario),
),
        #acquisition_maximizer= LocalSearch(configspace=model.configspace)


    )
    # define a callback to keep track of optimization?

    incumbent = smac.optimize()
    #print(mu_dict)

    # Get cost of default configuration
    default_cost = smac.validate(model.configspace.get_default_configuration())
    print(f"Default cost: {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")

    # Let's plot it too
    # plot_contour(smac.runhistory, incumbent)
    save_dir = "visualisations/with_noise/"

    intensifer = smac.intensifier
    # plot_contour(intensifier=intensifer, incumbent=incumbent, runhistory=smac.runhistory, model=model,
    #              save_dir=save_dir)

    plot_progress(smac)
    # save the results

