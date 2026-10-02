import time

import numpy as np

from Matrices import *


class NoPenalty:
    """Plain (un-regularised) logistic regression."""

    def __call__(self, W):
        return 0.0

    def derivative(self, W):
        return np.zeros_like(W)


class RidgePenalty:
    """
    L2 penalty:  lambda * sum(theta^2)      ->  gradient: 2 * lambda * theta
    Row 0 of W is the intercept (X's first column is all ones), and the
    intercept is NOT penalised (standard practice: shrinking the bias only
    hurts the fit and does not reduce model complexity).
    """

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, theta):
        return self.lambda_ * np.sum(np.square(theta[1:]))

    def derivative(self, theta):
        grad = 2 * self.lambda_ * theta
        grad[0] = 0.0  # leave the intercept alone
        return grad


class LassoPenalty:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, theta):
        return self.lambda_ * np.sum(np.square(theta[1:]))  # skip intercept

    def derivative(self, theta):
        return self.lambda_ * np.sign(theta)


class LogisticRegression(object):
    """
    Softmax regression trained with batch / mini-batch / stochastic gradient descent.

    X must already contain the intercept column (first column of ones), shape (m, n).
    Y is one-hot (m, k); a 1-D label vector is also accepted and one-hot encoded for you.
    W has shape (n, k).
    """

    def __init__(
        self,
        k,
        method="batch",
        alpha=0.1,
        epoch=5000,
        use_penalty=False,
        lambda_=0.01,
        verbose=False,
        batch_size=32,
        penalty_type="noP",
        rng=np.random.default_rng(42),
    ):
        self.k = k  # number of classes
        self.method = method  # "batch" | "minibatch" | "sto"
        self.alpha = alpha  # learning rate
        self.epoch = epoch
        self.use_penalty = use_penalty  # Task 2: choose ridge on/off
        self.lambda_ = lambda_
        self.verbose = verbose
        self.penalty_type = penalty_type
        self.rng = rng
        self.batch_size = batch_size

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X, dtype=np.float32)
        X = np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))  # add y_intercept
        y = np.asarray(y)

        if y.ndim == 1:
            y = self.one_hot(y)
        m = X.shape[0]

        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val, dtype=np.float32)
            X_val = np.hstack(
                (np.ones(X_val.shape[0]).reshape(-1, 1), X_val)
            )  # add y_intercept
            y_val = np.asarray(y_val)

        self.theta = self.rng.normal(
            0, 0.01, size=(X.shape[-1], self.k)
        )  # small random init
        self.losses = []

        start = time.time()
        best_val = None
        for ep in range(self.epoch):
            # reshuffle once per epoch
            idx = self.rng.permutation(m)
            X_shuf, Y_shuf = X[idx], y[idx]

            if self.method == "sto":
                for i in range(m):
                    X_method_train = X_shuf[i : i + 1]  # (1,n)
                    y_method_train = Y_shuf[i : i + 1]  # (1,k)
                    self._train(X_method_train, y_method_train, ep)

            elif self.method == "minibatch":
                for i in range(0, m, self.batch_size):
                    end = i + self.batch_size
                    X_method_train = X_shuf[i:end, :]
                    y_method_train = Y_shuf[i:end]
                    self._train(X_method_train, y_method_train, ep)

            elif self.method == "batch":
                X_method_train = X_shuf
                y_method_train = Y_shuf
                self._train(X_method_train, y_method_train, ep)

            else:
                raise ValueError(
                    'Method must be one of the following: "batch", "minibatch" or "sto".'
                )

            train_pred = self.predict(X)
            y_ = np.argmax(y, axis=1)  # convert back to original y
            train_epoch_acc = accuracy(y_, train_pred)

            if X_val is not None and y_val is not None:
                if y_val.ndim > 1:
                    y_val_ = np.argmax(y_val, axis=1)
                else:
                    y_val_ = y_val

                val_pred = self.predict(X_val)
                val_epoch_acc = accuracy(y_val_, val_pred)

                if best_val is not None and np.allclose(val_epoch_acc, best_val):
                    break
                best_val = val_epoch_acc
            else:
                val_epoch_acc = None

            if self.verbose and ep % 500 == 0:
                msg = f"Epoch {ep}: loss={self.losses[-1]:.4f}, train_acc={train_epoch_acc:.4f}"
                if X_val is not None and y_val is not None:
                    msg += f", val_acc={val_epoch_acc:.4f}"
                print(msg)

        self.train_time = time.time() - start
        self.final_loss = self.gradient(X, y)[0]  # loss on the WHOLE training set

        # final train accuracy
        self.train_acc = accuracy(np.argmax(y, axis=1), self.predict(X))

        # final validation accuracy
        self.val_acc = None
        self.final_val_loss = None
        y_val_true = y_val
        if X_val is not None and y_val is not None:
            if y_val.ndim > 1:
                y_val_true = np.argmax(y_val, axis=1)
            else:
                y_val_true = y_val

            if y_val.ndim < 2:
                y_val = self.one_hot(y_val)
            self.final_val_loss = self.gradient(X_val, y_val)[0]

            self.val_acc = accuracy(y_val_true, self.predict(X_val))

        return self.evaluate_all(
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val_true,
        )

    def evaluate_individual(self, X=None, y=None, loss=None):
        """
        Evaluates metrics for any single dataset split.
        """
        if X is None or y is None:
            return None

        y_true = np.argmax(y, axis=1) if y.ndim > 1 else np.asarray(y)
        y_pred = self.predict(X)
        k = self.k if isinstance(self.k, int) else len(self.k)

        return {
            "loss": loss,
            "acc": accuracy(y_true, y_pred),
            "macro_precision": macro_precision(y_true, y_pred, k),
            "macro_recall": macro_recall(y_true, y_pred, k),
            "macro_f1": macro_f1(y_true, y_pred, k),
            "weighted_precision": weighted_precision(y_true, y_pred, k),
            "weighted_recall": weighted_recall(y_true, y_pred, k),
            "weighted_f1": weighted_f1(y_true, y_pred, k),
            "per_class": {
                f"class_{c}": {
                    "precision": precision(y_true, y_pred, c),
                    "recall": recall(y_true, y_pred, c),
                    "f1": f1(y_true, y_pred, c),
                    "support": support(y_true, c),
                }
                for c in range(k)
            },
            "report_df": report(y_true, y_pred, k),
        }

    def evaluate_all(
        self,
        X=None,
        y=None,
        X_val=None,
        y_val=None,
        X_test=None,
        y_test=None,
        train_loss=None,
        val_loss=None,
        test_loss=None,
    ):
        """
        Evaluates metrics for train, validation, and test sets independently.
        """
        # Fetch default model losses if available and not passed explicitly
        if train_loss is None:
            train_loss = getattr(self, "final_loss", None)
        if val_loss is None:
            val_loss = getattr(self, "final_val_loss", None)

        train_res = self.evaluate_individual(X, y, train_loss)
        val_res = self.evaluate_individual(X_val, y_val, val_loss)
        test_res = self.evaluate_individual(X_test, y_test, test_loss)

        def _get(res, key):
            return res[key] if res is not None else None

        return {
            # --- Train Metrics ---
            "train_loss": _get(train_res, "loss"),
            "train_acc": _get(train_res, "acc"),
            "train_macro_precision": _get(train_res, "macro_precision"),
            "train_macro_recall": _get(train_res, "macro_recall"),
            "train_macro_f1": _get(train_res, "macro_f1"),
            "train_weighted_precision": _get(train_res, "weighted_precision"),
            "train_weighted_recall": _get(train_res, "weighted_recall"),
            "train_weighted_f1": _get(train_res, "weighted_f1"),
            "train_per_class": _get(train_res, "per_class"),
            "train_report_df": _get(train_res, "report_df"),
            # --- Validation Metrics ---
            "val_loss": _get(val_res, "loss"),
            "val_acc": _get(val_res, "acc"),
            "val_macro_precision": _get(val_res, "macro_precision"),
            "val_macro_recall": _get(val_res, "macro_recall"),
            "val_macro_f1": _get(val_res, "macro_f1"),
            "val_weighted_precision": _get(val_res, "weighted_precision"),
            "val_weighted_recall": _get(val_res, "weighted_recall"),
            "val_weighted_f1": _get(val_res, "weighted_f1"),
            "val_per_class": _get(val_res, "per_class"),
            "val_report_df": _get(val_res, "report_df"),
            # --- Test Metrics ---
            "test_loss": _get(test_res, "loss"),
            "test_acc": _get(test_res, "acc"),
            "test_macro_precision": _get(test_res, "macro_precision"),
            "test_macro_recall": _get(test_res, "macro_recall"),
            "test_macro_f1": _get(test_res, "macro_f1"),
            "test_weighted_precision": _get(test_res, "weighted_precision"),
            "test_weighted_recall": _get(test_res, "weighted_recall"),
            "test_weighted_f1": _get(test_res, "weighted_f1"),
            "test_per_class": _get(test_res, "per_class"),
            "test_report_df": _get(test_res, "report_df"),
        }

    def _train(self, X, y, i):
        loss, grad = self.gradient(X, y)
        self.losses.append(loss)
        # update theta
        self.theta = self.theta - self.alpha * grad
        if self.verbose and i % 500 == 0:
            print(f"Loss at iteration {i}: {loss:.4f}")

    # gradient
    def gradient(self, X, Y):
        """
        loss = -(1/m) * sum(Y * log(h)) + penalty(W)
        grad = (1/m) * X^T (h - Y)       + penalty'(W)
        Both are averaged over m so alpha / lambda do not depend on batch size.
        """
        if self.penalty_type == "noP":
            self.penalty = NoPenalty()
        else:
            self.penalty = RidgePenalty(self.lambda_)

        m = X.shape[0]
        h = self.h_theta(X, self.theta)
        loss = -np.sum(Y * np.log(np.clip(h, 1e-12, 1.0))) / m + self.penalty(
            self.theta[1:]
        )
        error = h - Y
        grad = self.softmax_grad(X, error) / m

        penalty_grad = self.penalty.derivative(self.theta[1:])

        # no regularization for intercept
        penalty_grad_full = np.vstack((np.zeros_like(self.theta[:1]), penalty_grad))

        grad = grad + penalty_grad_full
        return loss, grad

    def softmax(self, theta_t_x):
        z = theta_t_x - np.max(
            theta_t_x, axis=1, keepdims=True
        )  # stability: avoids exp overflow
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        """
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.softmax(X @ W)

    # predict
    def predict_proba(self, X):
        return self.h_theta(np.asarray(X, dtype=np.float32), self.theta)

    def predict(self, X_test):
        return np.argmax(self.predict_proba(X_test), axis=1)

    def one_hot(self, y):
        Y = np.zeros((len(y), self.k))
        Y[np.arange(len(y)), np.asarray(y, dtype=int)] = 1
        return Y
