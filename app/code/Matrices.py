import numpy as np
import pandas as pd


def counts(y, ypred, c):
    y, ypred = np.asarray(y), np.asarray(ypred)
    tp = np.sum((ypred == c) & (y == c))
    fp = np.sum((ypred == c) & (y != c))
    fn = np.sum((ypred != c) & (y == c))
    return tp, fp, fn


def accuracy(y, ypred):
    y, ypred = np.asarray(y), np.asarray(ypred)
    return np.sum(y == ypred) / len(y)


def precision(y, ypred, c):
    tp, fp, _ = counts(y, ypred, c)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 0 when class c never occurs


def recall(y, ypred, c):
    tp, _, fn = counts(y, ypred, c)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 0 when class c never occurs


def f1(y, ypred, c):
    p, r = precision(y, ypred, c), recall(y, ypred, c)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0  # 0 when class c never occurs


def support(y, c):
    return int(np.sum(np.asarray(y) == c))  # how many TRUE samples of class c


def macro_precision(y, ypred, k):
    return np.mean([precision(y, ypred, c) for c in range(k)])


def macro_recall(y, ypred, k):
    return np.mean([recall(y, ypred, c) for c in range(k)])


def macro_f1(y, ypred, k):
    return np.mean([f1(y, ypred, c) for c in range(k)])


def weights(y, k):
    return np.array([support(y, c) for c in range(k)]) / len(y)


def weighted_precision(y, ypred, k):
    return np.sum(weights(y, k) * [precision(y, ypred, c) for c in range(k)])


def weighted_recall(y, ypred, k):
    return np.sum(weights(y, k) * [recall(y, ypred, c) for c in range(k)])


def weighted_f1(y, ypred, k):
    return np.sum(weights(y, k) * [f1(y, ypred, c) for c in range(k)])


def report(y, ypred, k):
    """Per-class table + accuracy + macro/weighted averages."""
    rows = {
        f"class {c}": [
            precision(y, ypred, c),
            recall(y, ypred, c),
            f1(y, ypred, c),
            support(y, c),
        ]
        for c in range(k)
    }
    n = len(y)
    rows["accuracy"] = [np.nan, np.nan, accuracy(y, ypred), n]
    rows["macro avg"] = [
        macro_precision(y, ypred, k),
        macro_recall(y, ypred, k),
        macro_f1(y, ypred, k),
        n,
    ]
    rows["weighted avg"] = [
        weighted_precision(y, ypred, k),
        weighted_recall(y, ypred, k),
        weighted_f1(y, ypred, k),
        n,
    ]
    return pd.DataFrame(rows, index=["precision", "recall", "f1-score", "support"]).T
