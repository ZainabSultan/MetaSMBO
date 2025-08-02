
import json
import argparse
from pathlib import Path
import numpy as np

# HERE = "/home/zainab/UniFreiburg/Semester4/AutoML_finalexam/automl-project-summer23-sumibo/"
HERE = Path(__file__).resolve().parent.parent

from matplotlib import pyplot as plt

def load_runhistory(path):
    with open(path) as f:
        rh_dict = json.load(f)
        print(rh_dict["stats"])
    return rh_dict

json_file =[]




def get_hp(hp_name:str, json_file_dict):
    initial_design_cap = 37
    num_trials = json_file_dict["stats"]["finished"]

    configs = json_file_dict["configs"]
    data = json_file_dict["data"]
    config_hp_values = []

    # for i in range(initial_design_cap, num_trials):
    for i in range(1, initial_design_cap):

        str_i = str(i) # to access key
        config_cost = data[i][4]
        #print(data[i])
        config_hp_value = configs[str_i][hp_name]
        config_hp_values.append(config_cost)
    return config_hp_values, np.arange(initial_design_cap, num_trials)

def plot_dist_samples():
    exp_c=0
    everything_values_hist=[]
    for exp in experiments:
        paths = []
        all_seeds_values = []
        all_time_values=[]
        # plt.figure()

        width=1
        bins = np.linspace(-10, 10, 30)

        [[0,]]
        x_offsets = np.linspace(-width, width, len(experiments), endpoint=False)

        for i in range(exp[2]):

            path =  str(HERE) +"/"+ "SMAC3Output"+ "/"+ exp[0] +"/"+ str(i) +"/"+ "runhistory.json"
            paths.append(path)


        for p in range(len(paths)):
            path = paths[p]
            run_history_dict = load_runhistory(path)
            values, time = get_hp(args.hp, run_history_dict)
            all_seeds_values.extend(values)
            all_time_values.extend(time)
        everything_values_hist.append(1 - np.array(all_seeds_values).flatten())

        # plt.scatter(all_time_values, all_seeds_values, label=exp[1], s=3.3)
        print(bins[exp_c])
    print(len(everything_values_hist))
<<<<<<< HEAD
    bin_boundaries = np.linspace(0.1, 0.7, 7)
    plt.hist(everything_values_hist,bins = bin_boundaries, label=[exp[1] for exp in experiments],
             color=[exp[-1] for exp in experiments])
=======

    plt.hist(everything_values_hist,bins = 10, label=[exp[1] for exp in experiments])
    #plt.xticks(0.1)
>>>>>>> origin/zainab_

        #plt.ylabel(args.hp+" value")
    plt.ylabel("Frequency")
    plt.xlabel("Accuracy")

    plt.legend(loc='upper right')

    plt.grid()
    #plt.title("initial sampling in "+ "three different approaches")
    plt.title("Histogram over quality of samples from initial design (Sobol)", fontweight='bold')
    plt.savefig(args.hp+" quality of initial samples sampling in "+ "three different approaches"+".png")
    plt.show()


def plot_quality_of_dists(n_bins):
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.
    #axs[0].hist(dist1, bins=n_bins)
    #axs[1].hist(dist2, bins=n_bins)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="plptting"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="p_gp_0.8",
        help="Path to directory containing the dataset",
    )
    parser.add_argument("--seed_range", type=int, default=2)
    parser.add_argument("--hp", type=str, default="batch_size")


    args = parser.parse_args()

    exp_name = args.exp_name

    if exp_name == "baselinefinal":
        exp_name="Baseline"
    elif exp_name == "p_gp_0.8":
        exp_name = "Gentle Pruning (0.8)"
    elif exp_name == "vanilla_box_pruning":
        exp_name="Box"
    elif exp_name=="reduce_lpi_v5":
        exp_name="reduced dim (LPI)"

    experiments = [
        ('baselinefinal', 'baseline', 2, 'tab:red'),
        ('reduce_lpi_v5', 'reduced dim (LPI)', 2, 'tab:orange'),
        ('vanilla_box_pruning', 'box', 2, 'tab:green'),
        ('p_gp_0.8', 'gentle pruning (0.8)', 2, 0, 'tab:blue'),
        #('prior', 'prior', 2, , 'tab:purple'),
        # ('warmstart_v4', 'warmstart greedy', 0),
        #('warmstart_allin', 'warmstart all-in', 2)
    ]

    n_bins = 10
    x = np.random.randn(1000, 3)

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # ax0, ax1, ax2, ax3 = axes.flatten()
    #
    # colors = ['red', 'tan', 'lime']
    # ax0.hist(x, n_bins, histtype='bar', color=colors, label=colors)
    # ax0.legend(prop={'size': 10})
    # ax0.set_title('bars with legend')
    # plt.show()


    plot_dist_samples()



    #fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

    # We can set the number of bins with the *bins* keyword argument.








