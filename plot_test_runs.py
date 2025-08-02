from matplotlib import pyplot as plt
import numpy as np
from extract_smac_data import TIME_ARRAY, smac_val_test_results_to_np

# time = np.linspace(1000, 21600, 207)
time = TIME_ARRAY
std_scale = 0.4

# experiments to plot with their smac names and how to show in the plot (smac_name, plot_name, first_index)
# first index is to keep only the last config from warmstarting in the data to plot
# doing this automatically is nicer but a bit tidious to implement ^^
# color can be specified for consistency
experiments = [
    # ('baseline_w_smac_fix', 'baseline', 0, 'tab:red'),
    # ('reduce_lpi_v5', 'reduced dim (LPI)', 0, 'tab:orange'),
    # ('vanilla_box_pruning', 'box', 0, 'tab:green'),
    ('outside_the_box_extended_01', 'bo1x', 0, 'tab:green'),
    # ('p_gp_0.8', 'gentle pruning (0.8)', 0, 'tab:blue'),
    # ('prior', 'prior', 0, 'tab:purple')
    # ('warmstart_v4', 'warmstart greedy', 0),
    # ('warmstart_allin', 'warmstart all-in', 0)
]


# Plotting
plt.figure(figsize=(6, 5))  # Adjust the figure size as needed

# # for test accuracies only
# for smac_name, exp, first_index, color in experiments:
#     val, test = smac_val_test_results_to_np(smac_name, time, first_index)
#     print(f'{smac_name}: {test.shape}')
#     mean_test = test.mean(axis=0)
#     std_test = test.std(axis=0) * std_scale
#     # plot experiment data
#     plt.plot(time, mean_test, color=color, label=exp)
#     plt.fill_between(time, mean_test - std_test, mean_test + std_test, color=color, alpha=0.2)


# for validation vs test accuracy
for smac_name, exp, first_index, color in experiments:
    val, test = smac_val_test_results_to_np(smac_name, time, first_index)
    mean_test = test.mean(axis=0)
    mean_val = val.mean(axis=0)
    line = plt.plot(time, mean_test, color=color, label=exp + ' (test)')[0]
    plt.plot(time, mean_val, color=color, linestyle='--', label=exp + ' (val)')



plt.xlabel('Walltime [s]')
plt.ylabel('accuracy')
plt.ylim(0.5, None)
# plt.ylim(0.4, None)
plt.legend()
plt.grid()
plt.show()
