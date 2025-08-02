import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class Prune:
    def __init__(self, datasets_x, datasets_y, search_space=None, potetntial_thres=0, model_type="xgboost"):

        # need to fit a gaussian process to estimate response function
        self.predictors = []

        if model_type == "xgboost":
            fitter_func = fit_xgboost
        elif model_type  =="rf":
            fitter_func = fit_RF
        elif model_type =="gp":
            fitter_func = fit_gaussian
        else:
            print("were futting a RF bc you specified it wrong")
            fitter_func = fit_RF
        for i in range(len(datasets_x)):
            model = fitter_func(datasets_x[i], datasets_y[i])
            self.predictors.append(model)
        #print(self.predictors[0].predict(datasets_x[0]), "SHOULDNT BE  ZERO")
        # self.gp_model = fit_gaussian(datasets_x, datasets_y)
        #self.best_performance = np.min(datasets_y)
        self.search_space = search_space
        self.configs_seen =[]
        # this holds the current best cost for each model
        self.current_best = np.repeat(np.finfo(float).max, len(self.predictors))
        self.potetntial_thres = potetntial_thres
        self.time=0

    def response_function_evaluation(self, hp_setting):
        predicted_performance = np.array([])
        hp_setting = np.nan_to_num(hp_setting, nan=-1) # impute the conditional HPs
        for model in self.predictors:

            mu = model.predict(hp_setting)
            predicted_performance = np.append(predicted_performance,mu)
        return predicted_performance

    def update_current_best(self, history):

        for config in history:
            config = np.nan_to_num(config, nan=-1)
            if not any(np.array_equal(config, prev_configs) for prev_configs in self.configs_seen):
                #config = history[config]
                # we havent seen it before, so we evaluate and compare
                for i in range(len(self.predictors)):
                    model = self.predictors[i]
                    mu= model.predict(config.reshape(1,-1))
                    curr_best = self.current_best[i]
                    if mu < curr_best:
                        self.current_best[i] = mu
                self.configs_seen.append(config)


    def prune(self, X, history):
        # partition search space into a grid
        # evaluate the centeres of the grid boxes
        # if improvement is 0, remove it from set
        pruned_configs = []
        #print(X, "unpruned")
        indices_selected=[]
        idx=0
        potential_array =[]
        #print(len(X), "this is the length of the Xs, shouldnnt be zero...")
        for config in X:
            pred = self.response_function_evaluation(config.reshape(1,-1))
            self.update_current_best(history)
            potential = self.current_best - pred
            #print(potential, config, pred)
            potential = np.sum(potential)
            potential_array.append(potential)
            #if potential >= self.potetntial_thres:
                #pruned_configs.append(config)
                #indices_selected.append(idx)
            #else:
                #print("pruned this:", config)
            idx+=1
        sorted_indices = np.argsort(np.array(potential_array))
        percentile = 0.2
        num_selected = int(percentile * len(X))
        indices_selected = sorted_indices[-num_selected:]
        #print(indices_selected, "check this with potentials")
        pruned_configs = X[indices_selected]
        #print(pruned_configs, "PRUNED")
        #print(len(X), len(pruned_configs), "fll vs pruned second val should be less")
        return np.array(pruned_configs), indices_selected

    def get_centre_points(self):
        # retreive centre points for evlautation
        space = self.config_space.space
        centre_by_hp = dict.fromkeys(space.keys())
        for hp_name, hp_bounds in space:
            interval = hp_bounds[1] - hp_bounds[0]


def fit_gaussian(X, y):
    # Sample data: hyperparameters and corresponding performance
    # hyperparameters = [...]  # List of hyperparameters
    # performance = [...]  # Corresponding performance metric
    #
    # # Convert lists to numpy arrays
    # X = np.array(hyperparameters)
    # y = np.array(performance)

    # Define the kernel (Squared Exponential kernel)
    kernel = RBF()

    # Create a Gaussian Process Regressor with the defined kernel
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    # Fit the Gaussian Process model to the data
    gp_model = gp_model.fit(X, y)

    return gp_model

    # Predict performance for new hyperparameters (for example)
    # new_hyperparameters = np.array([[param1_value, param2_value, ...]])  # Replace with actual values
    # predicted_performance, _ = gp_model.predict(new_hyperparameters, return_std=True)

    # print("Predicted Performance:", predicted_performance)


def fit_RF(X,y):

    regr=RandomForestRegressor()
    rf = regr.fit(X, y)
    return rf

def fit_xgboost(X,y):
    model = XGBRegressor()
    model =model.fit(X,y)
    return model
