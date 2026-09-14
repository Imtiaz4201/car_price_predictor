import numpy as np
import matplotlib.pyplot as plt
import os, sys
import datetime
import json


def feature_importance(coefs, feature_names=None, top_n=None):
    """
    Plots feature importance based on the magnitude of coefficients.

    IMPORTANT (per assignment note): this assumes input features are on
    the same scale (i.e. were scaled, e.g. with StandardScaler, before
    fitting). Otherwise, coefficient magnitude is not a fair measure of
    importance.

    Parameters
    ----------
    feature_names : list of str, optional
        Names of the features (excluding intercept). If None, generic
        names ('theta_1', 'theta_2', ...) are used.
    top_n : int, optional
        If given, only the top_n most important features are shown.
    """
    if feature_names is None:
        feature_names = [f"theta_{i+1}" for i in range(len(coefs))]

    assert len(feature_names) == len(
        coefs
    ), "feature_names length must match number of coefficients (excluding intercept)"

    importance = np.abs(coefs)
    order = np.argsort(importance)[::-1]  # descending

    if top_n is not None:
        order = order[:top_n]

    sorted_names = [feature_names[i] for i in order]
    sorted_coefs = coefs[order]

    plt.figure(figsize=(8, max(4, 0.4 * len(sorted_names))))
    colors = ["#d62728" if c < 0 else "#1f77b4" for c in sorted_coefs]
    plt.barh(sorted_names, sorted_coefs, color=colors)
    plt.xlabel("Coefficient value")
    plt.title("Feature Importance (based on coefficient magnitude)")
    plt.gca().invert_yaxis()
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.show()


def save_BT_results(model_parms, final_train_mse_log, final_train_r2_log,final_train_mse_exp,final_train_r2_exp, idx):
    res = {
        "model_info": model_parms,
        "mse_log": float(final_train_mse_log),
        "r2_log": float(final_train_r2_log),
        "mse_exp": float(final_train_mse_exp),
        "r2_exp": float(final_train_r2_exp),
        "datetime": datetime.datetime.now().isoformat(),
    }

    path = os.path.join(os.getcwd(), "saved_models")
    # create folder if not exists
    if not os.path.exists(path):
        os.makedirs(path)

    tr_file_path = os.path.join(path, "best_model_results"+str(idx)+".json")

    if os.path.exists(tr_file_path):
        with open(tr_file_path, "r") as f:
            tr_file = json.load(f)
    else:
        tr_file = []

    tr_file.append(res)

    with open(tr_file_path, "w") as f:
        json.dump(tr_file, f, indent=4)

def predict(X, theta):
    """
    X -> (sample, features)
    theta -> (sample,)
    """
    y_hat = (X @ theta.reshape(-1,1)).flatten()  # (batch,)
    return y_hat

