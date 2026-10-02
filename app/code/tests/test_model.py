import sys
from pathlib import Path

# Add parent directory (app/code) to sys.path so Model can be imported
CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
from Model import LogisticRegression


N_SAMPLES = 20
N_FEATURES = 5
N_CLASSES = 4


def make_dummy_data(n_samples=N_SAMPLES, n_features=N_FEATURES, k=N_CLASSES, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.integers(0, k, size=n_samples)
    return X, y


def make_model(method="batch", batch_size=None, rng=None):
    rng = rng or np.random.default_rng(0)
    return LogisticRegression(
        k=N_CLASSES,
        method=method,
        alpha=0.01,
        epoch=5,
        use_penalty=False,
        lambda_=0.0,
        verbose=False,
        batch_size=batch_size,
        penalty_type="noP",
        rng=rng,
    )


def test_model_accepts_expected_input():
    """Model should fit without error on correctly-shaped synthetic input."""
    X, y = make_dummy_data()
    model = make_model(method="batch")
    model.fit(X, y)

    # Asserts model learned a theta matrix with k columns
    assert model.theta.shape[1] == N_CLASSES


def test_model_output_shape():
    """predict() should return one label per sample, within the valid class range."""
    X, y = make_dummy_data()
    model = make_model(method="batch")
    model.fit(X, y)

    # Prepend bias/intercept column of 1s to match self.theta shape (6 rows)
    bias = np.ones((X.shape[0], 1))
    X_test = np.hstack([bias, X])

    preds = model.predict(X_test)
    preds = np.asarray(preds)

    assert preds.shape[0] == N_SAMPLES
    assert preds.min() >= 0 and preds.max() < N_CLASSES