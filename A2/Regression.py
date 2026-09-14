import numpy as np
from sklearn.model_selection import KFold
import mlflow
from sklearn.base import clone

import sys
import os
import json
import joblib

sys.path.append(os.path.abspath("."))
from Formulas import *
from Utils import save_BT_results
from PipeLines import *


class LinearRegression(object):
    """ """

    def __init__(
        self,
        regularization,
        lr=0.001,
        method="batch",
        num_epochs=100,
        batch_size=16,
        n_splits=5,
        init_method="zeros",
        momentum=0.0,
        use_momentum=False,
        rng = None
    ):
        self.cv = KFold(n_splits, shuffle=True, random_state=42)
        self.regularization = regularization
        self.lr = lr
        self.method = method
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_method = init_method
        self.momentum = momentum
        self.use_momentum = use_momentum
        self.rng = rng
    # Weight initialization
    def _initialize_theta(self, n_features, m_samples):
        """
        n_features : number of columns in X, number of theata needed
        m_samples : number of samples in the current training fold
        """
        if self.init_method == "xavier":
            lower, upper = -(1.0 / np.sqrt(m_samples)), (1.0 / np.sqrt(m_samples))
            theta = lower + self.rng.random(n_features) * (upper - lower)
        else:
            theta = np.zeros(n_features)

        return theta

    # Training
    def fit(self, X_train, y_train, use_poly=False, k_features=11, idx_=None):
        """
        X_train : shape -> (sample, features)
        y_train : shape -> (sample,)
        """
        self.kfold_scores = list()
        self.kfold_r2_scores = list()
        self.fold_pipelines = list()
        self.val_loss_old = np.inf

        for fold, (train_index, val_idx) in enumerate(self.cv.split(X_train)):
            X_tr_raw = X_train.iloc[train_index]
            X_val_raw = X_train.iloc[val_idx]
            y_cross_train = y_train[train_index]
            y_cross_val = y_train[val_idx]

            ct_fold, sk_fold = clone(ct), sk(k_features)

            X_tr = ct_fold.fit_transform(X_tr_raw, y_cross_train)
            X_tr = sk_fold.fit_transform(X_tr, y_cross_train)

            X_val = ct_fold.transform(X_val_raw)
            X_val = sk_fold.transform(X_val)

            poly_fold = None
            if use_poly:
                poly_fold = clone(poly_pipeline)
                X_tr = poly_fold.fit_transform(X_tr)
                X_val = poly_fold.transform(X_val)

            # add intercept
            X_cross_train = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
            X_cross_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])

            self.fold_pipelines.append(
                {"ct": ct_fold, "sk": sk_fold, "poly": poly_fold}
            )

            # inti theta with zeros or xavier
            self.theta = self._initialize_theta(
                n_features=X_cross_train.shape[1], m_samples=X_cross_train.shape[0]
            )  # (F,)

            # momentum book keeping
            self.prev_step = np.zeros_like(self.theta)  # (features, 1)

            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True) as fold_run:
                params = {
                    "method": self.method,
                    "lr": self.lr,
                    "reg": type(self).__name__,
                    "use_poly": use_poly,
                    "init_method": self.init_method,
                    "use_momentum": self.use_momentum,
                    "momentum": self.momentum if self.use_momentum else 0.0,
                }

                fold_dir = os.path.join(
                    "saved_models", "fold_pipelines"+str(idx_), fold_run.info.run_id
                )
                os.makedirs(fold_dir, exist_ok=True)
                joblib.dump(ct_fold, os.path.join(fold_dir, "ct.pkl"))
                joblib.dump(sk_fold, os.path.join(fold_dir, "sk.pkl"))
                if poly_fold is not None:
                    joblib.dump(poly_fold, os.path.join(fold_dir, "poly.pkl"))
                mlflow.log_artifacts(fold_dir, artifact_path="preprocessors")
                mlflow.log_params(params=params)

                # training loops
                for epoch in range(self.num_epochs):
                    # shuffle X_cross_train and y_cross_train
                    idx = self.rng.permutation(X_cross_train.shape[0])
                    X_cross_train = X_cross_train[idx]
                    y_cross_train = y_cross_train[idx]

                    if self.method == "sto":
                        # stochastic
                        for i in range(len(X_cross_train)):
                            # get a single sample
                            X_method_train = X_cross_train[i].reshape(1, -1)  # (S,F)
                            y_method_train = y_cross_train[i]  # (S,)
                            # forward pass
                            train_loss = self.train_model(
                                X_method_train, y_method_train
                            )
                    elif self.method == "mini":
                        # mini batch
                        for i in range(0, len(X_cross_train), self.batch_size):
                            end_ = i + self.batch_size
                            X_method_train = X_cross_train[i:end_, :]
                            y_method_train = y_cross_train[i:end_]
                            train_loss = self.train_model(
                                X_method_train, y_method_train
                            )
                    else:
                        # full batch
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self.train_model(X_method_train, y_method_train)

                    mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                    # validation prediction
                    # val mse
                    y_val_hat_pred = self.predict(X_cross_val)
                    val_loss_new = mse(y_val_hat_pred, y_cross_val)
                    mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)

                    # val r2
                    val_r2_new = r2(y_val_hat_pred, y_cross_val)
                    mlflow.log_metric(key="val_r2", value=val_r2_new, step=epoch)

                    if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    self.val_loss_old = val_loss_new

                # show val loss after each epoch
                self.kfold_scores.append(val_loss_new)
                self.kfold_r2_scores.append(val_r2_new)
                print(f"Fold {fold}: {val_loss_new}")

    def fit_final(
        self, X_train, y_train, save_weight=True, model_name=None, idx_=None,use_poly=False
    ):
        """
        Train the final model on the complete training dataset.

        X_train : shape -> (samples, features)
        y_train : shape -> (samples,)
        """
        # add intercept
        X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        # Initialize theta
        self.theta = self._initialize_theta(
            n_features=X_train.shape[1], m_samples=X_train.shape[0]
        )

        # Momentum bookkeeping
        self.prev_step = np.zeros_like(self.theta)

        with mlflow.start_run(run_name="BEST_MODEL_TRAINING"):

            params = {
                "method": self.method,
                "lr": self.lr,
                "reg": type(self).__name__,
                "use_poly": use_poly,
                "init_method": self.init_method,
                "use_momentum": self.use_momentum,
                "momentum": self.momentum if self.use_momentum else 0.0,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
            }

            mlflow.log_params(params)

            # Training loop
            for epoch in range(self.num_epochs):

                idx = self.rng.permutation(X_train.shape[0])
                X_train_batch = X_train[idx]
                y_train_batch = y_train[idx]

                if self.method == "sto":

                    # Stochastic gradient descent
                    for i in range(len(X_train_batch)):

                        X_method_train = X_train_batch[i].reshape(1, -1)
                        y_method_train = y_train_batch[i]

                        train_loss = self.train_model(
                            X_method_train, y_method_train, save_weight, idx_
                        )

                elif self.method == "mini":

                    # Mini-batch gradient descent
                    for i in range(0, len(X_train_batch), self.batch_size):

                        end_ = i + self.batch_size

                        X_method_train = X_train_batch[i:end_, :]
                        y_method_train = y_train_batch[i:end_]

                        train_loss = self.train_model(
                            X_method_train, y_method_train, save_weight, idx_
                        )

                else:

                    # Full-batch gradient descent
                    X_method_train = X_train_batch
                    y_method_train = y_train_batch

                    train_loss = self.train_model(
                        X_method_train, y_method_train, save_weight, idx_
                    )

                # Log training loss
                mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Final training prediction
            y_train_pred_log = self.predict(X_train)

            final_train_mse_log = mse(y_train_pred_log, y_train)
            final_train_r2_log = r2(y_train_pred_log, y_train)

            res = {"mse": final_train_mse_log, "r2": final_train_r2_log}

            # Transform predictions and actual values back to original price scale

            y_train_pred_exp = np.exp(y_train_pred_log)
            y_train_exp = np.exp(y_train)
            # Metrics in original price scale
            final_train_mse_exp = mse(y_train_pred_exp, y_train_exp)
            final_train_r2_exp = r2(y_train_pred_exp, y_train_exp)

            # Save both scale results into JSON
            model_key = f"{model_name}_{self.use_momentum}_{self.momentum}_{self.method}_{self.lr}"

            save_BT_results(
                model_key,
                float(final_train_mse_log),
                float(final_train_r2_log),
                float(final_train_mse_exp),
                float(final_train_r2_exp),
                idx=idx_,
            )

            mlflow.log_metric("final_train_mse_log", final_train_mse_log)
            mlflow.log_metric("final_train_r2_log", final_train_r2_log)

            print(f"Final training MSE log: {final_train_mse_log:.4f}")
            print(f"Final training R2 log: {final_train_r2_log:.4f}")

            return res

    def train_model(self, X, y, saved_weight=False, idx=None):
        """
        X/input : shape -> (batch, features)
        y/label : shape -> (batch,)
        """
        y_hat = self.predict(X)  # (batch,)
        m = X.shape[0]

       # loss = 0.5 * np.sum((y_hat - y) ** 2)  # (1,1)

        e = y_hat - y  # (batch,)
        grads = (1 / m) * X.T @ e + self.regularization.derivation(
            self.theta
        )  # (features,)

        #  update theta
        step, self.theta = self.update_theta(
            self.lr, self.use_momentum, grads, self.theta, self.momentum, self.prev_step
        )

        self.prev_step = step  # (features, 1)

        if saved_weight:
            path = os.path.join(os.getcwd(), "saved_models")
            os.makedirs(path, exist_ok=True)

            weight_file = os.path.join(path, "weights" + str(idx) + ".json")

            latest_weights = {
                "bias": float(self._bias()),
                "coefficients": self._coef().tolist(),
            }

            with open(weight_file, "w") as f:
                json.dump(latest_weights, f, indent=4)

        return mse(y_hat, y)

    def update_theta(self, lr, use_momentum, grads, old_theta, momentum, prv_step):
        step = lr * grads  # (features, 1)
        if use_momentum:
            new_theta = old_theta - step + momentum * prv_step
        else:
            new_theta = old_theta - step

        return step, new_theta

    def predict(self, X):
        # X's shape (batch, F)
        y_hat = (X @ self.theta.reshape(-1, 1)).flatten()  # (batch,)
        return y_hat

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]


# Penalty classes
class LassoPenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.abs(theta))

    def derivation(self, theta):
        return self.l * np.sign(theta)


class RidgePenalty:
    def __init__(self, l):
        self.l = l

    def __call__(self, theta):
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta):
        return self.l * 2 * theta


class ElasticPenalty:
    def __init__(self, l=0.1, l_ratio=0.5):
        self.l = l
        self.l_ratio = l_ratio

    def __call__(self, theta):
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return l1_contribution + l2_contribution

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return l1_derivation + l2_derivation


class NoPenalty:
    """
    'Normal' regression, i.e. no regularization at all.
    Needed so that 'Normal' fits the same interface as Ridge/Lasso in Task 2's
    comparison loop (axis 1: polynomial, lasso, ridge, normal).
    """

    def __call__(self, theta):
        return 0

    def derivation(self, theta):
        return np.zeros_like(theta)
