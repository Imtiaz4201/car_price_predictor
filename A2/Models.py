import sys
import os
sys.path.append(os.path.abspath("."))

from Regression import *

class NormalLinearRegression(LinearRegression):
    def __init__(
        self,
        lr=0.001,
        method="batch",
        num_epochs=100,
        batch_size=16,
        n_splits=5,
        init_method="zeros",
        momentum=0,
        use_momentum=False,
        rng = None
    ):
        super().__init__(
            NoPenalty(),
            lr,
            method,
            num_epochs,
            batch_size,
            n_splits,
            init_method,
            momentum,
            use_momentum,
            rng
        )


class LassoLinearRegression(LinearRegression):
    def __init__(
        self,
        l=0.1,
        lr=0.001,
        method="batch",
        num_epochs=100,
        batch_size=16,
        n_splits=5,
        init_method="zeros",
        momentum=0,
        use_momentum=False,
        rng = None
    ):
        super().__init__(
            LassoPenalty(l),
            lr,
            method,
            num_epochs,
            batch_size,
            n_splits,
            init_method,
            momentum,
            use_momentum,
            rng
        )


class RidgeLinearRegression(LinearRegression):
    def __init__(
        self,
        l=0.1,
        lr=0.001,
        method="batch",
        num_epochs=100,
        batch_size=16,
        n_splits=5,
        init_method="zeros",
        momentum=0,
        use_momentum=False,
        rng = None
    ):
        super().__init__(
            RidgePenalty(l),
            lr,
            method,
            num_epochs,
            batch_size,
            n_splits,
            init_method,
            momentum,
            use_momentum,
            rng
        )
