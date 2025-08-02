from __future__ import annotations
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
import shap
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from dataclasses import dataclass


import pandas as pd
import numpy as np

from pathlib import Path
from dask.distributed import get_worker
from sklearn.model_selection import StratifiedKFold
from smac import Scenario, HyperparameterOptimizationFacade
from smac.runhistory import TrialValue
from smac.initial_design import AbstractInitialDesign
from smac.runhistory.dataclasses import TrialInfo
from torch.utils.data import DataLoader, Subset
from datasets import load_deep_woods
import torch
from cnn import Model
from torchvision.datasets import ImageFolder
import argparse
from prune_space_template import Data, get_metadata, HERE

METADATA_FILE = HERE / "metadata" / "deepweedsx_balanced-epochs-trimmed.csv"
DATA_PATH = HERE / "datasets"

import logging

logger = logging.getLogger(__name__)


def t_sne(metadata_dataframe: pd.DataFrame, num_seeds=5, thres=0.5, use_thres=True):
    # plots the t-sne

    metadata_dataframe = preprocess_data(metadata_dataframe)

    if use_thres:
        y = metadata_dataframe["cost"] < thres
    else:
        y = metadata_dataframe["cost"]
    x = metadata_dataframe.drop(columns=["cost"])  # input features
    complete_dataframe_list = []
    # columns that are not numerical, used for label encoder later...
    object_cols = ['use_BN']

    z_vals = []
    for seed in range(num_seeds):
        # repeating t_sne because the plots might be due to chance
        tsne = TSNE(n_components=2, verbose=1, random_state=seed)

        z = tsne.fit_transform(x)
        # store data for this run in dataframe
        df = pd.DataFrame()
        df["y"] = y
        df["comp-1"] = z[:, 0]
        df["comp-2"] = z[:, 1]
        df["seed"] = seed
        complete_dataframe_list.append(df)
    complete_dataframe = pd.concat(complete_dataframe_list, ignore_index=True)

    assert len(complete_dataframe.iloc[:, 0]) % num_seeds == 0  # shape sanity check

    g = sns.FacetGrid(complete_dataframe, col="seed", hue="y")
    g.map(sns.scatterplot, "comp-1", "comp-2", alpha=.7)
    if use_thres:
        g.add_legend()
    g.savefig("tsne_thres")


def feature_importance(metadata_dataframe: pd.DataFrame, technique: str, num_seeds=5, use_thres=False, thres=0.5):
    # technique is how to get the feature importance
    metadata_dataframe = preprocess_data(metadata_dataframe)
    if use_thres:
        y = metadata_dataframe["cost"] < thres
    else:
        y = metadata_dataframe["cost"]
    x = metadata_dataframe.drop(columns=["cost"])  # input features
    print(x.shape)
    if technique.lower() == "shap":
        # use the shapley values
        # TODO:  repeat multiple times and plot the variance + mean
        model = XGBRegressor(n_estimators=10, max_depth=10, learning_rate=0.001)
        # Fit the Model
        model.fit(x, y)
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)
        print(shap_values.shape, x.columns)
        print(np.abs(shap_values).mean(0))
        # set show to false to be able to save
        # because plt.show is automatically called
        # shap.summary_plot(shap_values, features=x, feature_names=x.columns, show=False)
        # plt.savefig('shap_feature_importance_plot.png')
        # plt.show() # clear buffer
        # shap.summary_plot(shap_values, features=x, feature_names=x.columns, plot_type="bar", show = False)
        # plt.savefig('shap_feature_importance_bar.png')


def plot_hp_vs_perf(metadata, hps):
    metadata_dataframe = preprocess_data(metadata)
    for hp in hps:
        mean_score = metadata.groupby(hp)["cost"].mean().reset_index()
        std_score = metadata.groupby(hp)["cost"].std().reset_index()
        print(mean_score)
        print(std_score)
        plt.errorbar(mean_score[hp], mean_score["cost"], yerr=std_score["cost"], fmt="o")
        plt.title(hp)
        plt.xlabel("value")
        plt.ylabel("cost")
        plt.savefig("{hp} vs performance.png".format(hp=hp))

        plt.show()


def preprocess_data(metadata_dataframe: pd.DataFrame, columns_to_drop=None, drop_conditional=True):
    # this function drops columns anc preprocesses the data
    #
    if columns_to_drop is None and drop_conditional:
        columns_to_drop = ["status", 'instance', 'budget', 'time', 'status', 'starttime', 'endtime',
                   "n_channels_conv_1", "n_channels_conv_2", "n_channels_fc_1", "n_channels_fc_2"]
    if columns_to_drop is None and not drop_conditional:
        columns_to_drop = ["status", 'instance', 'budget', 'time', 'status', 'starttime', 'endtime']

    metadata_dataframe = metadata_dataframe.drop(
        columns=columns_to_drop)
    print(metadata_dataframe.columns)
    metadata_dataframe["use_BN"] = LabelEncoder().fit_transform(metadata_dataframe["use_BN"])

    # in case of conditional HPs
    # metadata_dataframe = metadata_dataframe.fillna(0)
    return metadata_dataframe


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

    metadata_dataframe_ = get_metadata(args.metadata_file)
    # t_sne(metadata_dataframe_)
    # config:,config:global_avg_pooling,config:,config:,config:,config:,config:,config:,config:,config:,config:,scenario_seed,config:n_channels_fc_1,config:n_channels_fc_2
    hps = ["global_avg_pooling","learning_rate_init","n_channels_conv_0","n_channels_fc_0","n_conv_layers","n_fc_layers","use_BN"]
    #plot_hp_vs_perf(metadata_dataframe_, hps)
    #plt.show()
    # from scipy.stats import norm
    # normalized_accuracies = 1-metadata_dataframe_["cost"]
    # learning_rates = metadata_dataframe_["learning_rate_init"]
    # weighted_mean = np.sum(learning_rates * normalized_accuracies) / np.sum(normalized_accuracies)
    # print(weighted_mean)
    # weighted_std = np.sqrt(
    #     np.sum(normalized_accuracies * (learning_rates - weighted_mean) ** 2) / np.sum(normalized_accuracies))
    #
    # # Create a range of learning rates for visualization
    # x_range = np.linspace(min(learning_rates), max(learning_rates), 100)
    #
    # # Calculate the PDF of the weighted normal distribution
    # pdf_values = norm.pdf(x_range, weighted_mean, weighted_std)
    #
    # # Create a new figure
    # plt.figure(figsize=(10, 6))
    #
    # # Plot histogram of the learning rates
    # plt.hist(learning_rates, bins=10, density=True, alpha=0.6, color='blue', label='Learning Rates')
    #plt.scatter(metadata_dataframe_["batch_size"], 1-metadata_dataframe_["cost"])
    #
    # # Plot the fitted weighted normal distribution
    # plt.plot(x_range, pdf_values, 'r--', label='Fitted Weighted Normal')
    #
    # # Add labels and a legend
    # plt.xlabel('Learning Rate')
    # plt.ylabel('Density')
    # plt.title('Fitted Distribution Peaked Around High Accuracy Learning Rates')
    # plt.legend()
    #plt.show()

    #feature_importance(metadata_dataframe_, "shap")
    # tips = sns.load_dataset("tips")
    # print(tips.head())

    from ConfigSpace import NormalIntegerHyperparameter, BetaIntegerHyperparameter, NormalFloatHyperparameter
    p=NormalFloatHyperparameter(name="m", mu=1e-3,sigma=1, lower=0.00001,upper=1, log=True)
    po= BetaIntegerHyperparameter(name="l", alpha=2, beta=4, lower=2, upper=5)
    print(p)
