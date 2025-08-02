from visualisation.analyse import 

class feature_importance:
    def __init__(self, type,data):
        self.type = type
        self.data = data

    def feature_importance():
        if type == "shapley":



def shapley_feature_importance(metadata_dataframe: pd.DataFrame, technique: str,num_seeds=5, use_thres=False, thres=0.5):
# technique is how to get the feature importance
    metadata_dataframe = preprocess_data(metadata_dataframe)
    if use_thres:
        y = metadata_dataframe["cost"] < thres
    else:
        y = metadata_dataframe["cost"]
    x = metadata_dataframe.drop(columns=["cost"])  # input features
    if technique.lower() == "shap":
        # use the shapley values
        # TODO:  repeat multiple times and plot the variance + mean
        model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001)
        # Fit the Model
        model.fit(x, y)
        shap.initjs()
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x)



def preprocess_data(metadata_dataframe: pd.DataFrame, columns=None, drop_conditional=True):
    # this function drops columns anc preprocesses the data
    #
    if columns is None and drop_conditional:
        columns = ["status", 'instance', 'seed', 'budget', 'time', 'status', 'starttime', 'endtime',
                   "n_channels_conv_1","n_channels_conv_2","n_channels_fc_1","n_channels_fc_2"]
    if columns is None and not drop_conditional:
        columns = ["status", 'instance', 'seed', 'budget', 'time', 'status', 'starttime', 'endtime']

    metadata_dataframe = metadata_dataframe.drop(
        columns=columns)
    print(metadata_dataframe.columns)
    metadata_dataframe["use_BN"] = LabelEncoder().fit_transform(metadata_dataframe["use_BN"])

    # in case of conditional HPs
    # metadata_dataframe = metadata_dataframe.fillna(0)
    return metadata_dataframe

