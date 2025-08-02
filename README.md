# Improving SMAC using meta learning


The results and methodology of this project have been reported in our poster, it is available [here](https://docs.google.com/presentation/d/1xrqW7t2xgRURZNMXMYRSrohDhscch6HFZrE38uW5l14/edit?usp=sharing)

This repository contains:

1. Source code to infer a prior from the data and use it jointly with piBO
2. Source code to do search space pruning based on potential of configuration (based on work by Martin Witsuba et al titled Hyperparameter Search Space Pruning – A New Component for Sequential Model-Based Hyperparameter Optimization.)
   

### Get started

#### Conda
```bash
conda create -n automl python=3.10
conda activate automl
pip install -r requirements.txt
```

#### Venv

```bash
# Make sure you have python 3.8/3.9/3.10
python -V
python -m venv my-virtual-env
./my-virtual-env/bin/activate
pip install -r requirements.txt
```

#### SMAC
If you have issues installing SMAC,
follow the instructions [here](https://automl.github.io/SMAC3/main/1_installation.html).


### Data
You need to pre-download all the data required by running `python datasets.py`.

Stores by default in a `./data` directory. Takes under 20 seconds to download and extract.

## job_wrapper.sh

This script is a wrapper on top of job.sh, used to submit jobs to the cluster

## job.sh

Bash script that allocates appropriate resources to the script and runs it. 

## prune_space_template.py

This contains the template to run SMAC3 with hyperparameter space pruning based on work titled Hyperparameter Search Space Pruning – A New Component for Sequential Model-Based  Hyperparameter Optimization.

## prior_template.py

Infers a prior from the meta data, then use it to augment the acquisition function following the work of piBO

## warmstart_template.py

Runs the following experiments from the poster according to the flags:

--warmstart-smbo: will run the greedy version of warmstarting as

--warmstart-all-in: the all-in version

--box-pruning: the normal box-pruning with lower and upper bounds

--box-outside: the box-pruning that only prunes upper bounds (alternative approach)

--lpi-reduce-dim: reduces ConfigSpace dimension based on LPI

--fanova-reduce-dim: reduces ConfigSpace dimension based on fANOVA

## importance_utils.py / local_importance.py / global_importances.py

function to get pruned configs space when running experiment is in importance_utils.py. It interfaces local_importance.py for LPI and global_importances.py for fANOVA.

## box_pruning.py
Implementation of the box pruning for both --box-pruning and --box-outside.

## plot_test_runs.py / plot_lpi_vs_fanova.py / extract_smac_data.py

Files used for plots in the paper. extract_smac_data.py provides incumbent trajectories from a SMAC run as np.arrays.

## toy_env.py 

Contains a toy optimisation environment (Rosenbrock and weighted quadratic). Used for debugging and testing purposes as a white box function.

## Algorithms

This contains helper classes to the prior and pruning approach. 

* fit_distrbution.py: used to get the probability weights for catagorical data in the prior appraoch
* pruned_searchspace.py: implements pruning 
* local_search_pruned.py, pruned_expected_improvement.py : override SMAC to allow pruning

## visualisation

Contains some handy visualisations (interaction of hyperparameters and performance, data distributions) and scripts for analysis

## SMAC3_output

Results of all experiments
to interpret the naming convention of the pruning experiments:
* results of pruning are of the format {p}_{model_used}_{aggressiveness_of_pruning}
* model_used can be one of three: rf (Random forest), xg (XGboost), gp (gaussian process)
* aggressiveness of pruning could be 0.8 or 0.2 - referring to whether we keep the top 80% or 20% of the configurations sampled



