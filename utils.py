import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
 
from pytorch_tabnet.pretraining import TabNetPretrainer

def save2load_pretrain_model(model, save=True):
    if save:
        model.save_model('./test_pretrain')
    loaded_pretrain = TabNetPretrainer()
    loaded_pretrain.load_model('./test_pretrain.zip')
    return loaded_pretrain

        
def feature_importances(model, df, target_name):
    feature_name =[str(col) for col in df.columns if col!=target_name]
    print(len(feature_name))
    feat_imp = pd.DataFrame(model.feature_importances_, index=feature_name)
    feature_importance = feat_imp.copy()

    feature_importance["imp_mean"] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values("imp_mean")

    plt.tick_params(labelsize=18)
    plt.barh(feature_importance.index.values, feature_importance["imp_mean"])
    plt.title("feature_importance", fontsize=18)
    

def calculate_scores(true, pred):
    scores = {}
    scores = pd.DataFrame(
        {
            "R2": r2_score(true, pred),
            "MAE": mean_absolute_error(true, pred),
            "MSE": mean_squared_error(true, pred),
            "RMSE": np.sqrt(mean_squared_error(true, pred)),
        },
        index=["scores"],
    )
    return scores