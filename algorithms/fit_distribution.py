import numpy as np


def fit_continuous(hp_data, hp_performance):
    # fits beta, normal, uniform dists based on MLE
    # calucluates which is best fit + returns a dict
    possible_dits = []


import pandas as pd


def get_weights(hp_data, performance):

    num_cat = pd.unique(hp_data)
    # print(num_cat)
    perf_array = np.ones(len(num_cat))*0.1
    for i in range(len(num_cat)):
        cat = num_cat[i]
        print(cat, "this is ace")
        perf = performance[hp_data == cat]
        avrg = np.sum(perf) / len(perf)
        perf_array[i] = avrg
    perf_array = 1 - perf_array
    # print(perf_array)
    performance_values = np.array(perf_array)
    normalized_performance = (performance_values - np.min(performance_values)) / (
                np.max(performance_values) - np.min(performance_values))

    # Step 2: Calculate probabilities using normalized performance scores
    #probabilities = normalized_performance / np.sum(normalized_performance)
    #print(probabilities)

    k = 5  # Scaling factor
    x0 = 0.5  # Midpoint

    # Calculate probabilities using the sigmoid function
    probabilities = 1 / (1 + np.exp(-k * (normalized_performance - x0)))

    # Normalize probabilities to sum to 1
    #probabilities /= np.sum(probabilities)
    # print(probabilities)
    # Step 1: Scale performance values to a desired range
    #scaled_values = np.interp(performance_values, (0.0, 1.0), (1, 100))

    # Step 2: Convert scaled values into integer weights

    #integer_weights = np.round(scaled_values).astype(int)
    return probabilities

metadata_dataframe_ = pd.read_csv("/home/zainab/UniFreiburg/Semester4/AutoML_finalexam/automl-project-summer23-sumibo/metadata/deepweedsx_balanced-epochs-trimmed.csv")
w=get_weights(metadata_dataframe_["config:global_avg_pooling"].sort_values(ascending=False), metadata_dataframe_["cost"])
print(w)