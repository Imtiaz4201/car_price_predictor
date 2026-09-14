from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from functools import partial

"""
PipeLine
Involved processes:
- SimpleImputer
- StandardScaler
- TargetEncoder
- OneHotEncoder
- KFold
- SelectKBest
- ColumnTransformer
"""

num_impute_pipeline = Pipeline(
    steps=[
        ("median_imput", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
)
cat_impute_pipeline = Pipeline(
    steps=[
        ("cat_imput", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ]
)
target_enc_pipeline = Pipeline(
    steps=[
        (
            "target_enc",
            TargetEncoder(target_type="continuous", smooth="auto", random_state=42),
        ),
        ("scale", StandardScaler()),
    ]
)
scale_pipeline = Pipeline(steps=[("scale", StandardScaler())])

ct = ColumnTransformer(
    transformers=[
        ("num", num_impute_pipeline, ["mileage", "engine"]),
        ("cat", cat_impute_pipeline, ["seats"]),
        ("other_scale", scale_pipeline, ["year", "km_driven", "max_power"]),
        (
            "OHE",
            OneHotEncoder(handle_unknown="ignore"),
            ["fuel", "seller_type", "transmission", "owner"],
        ),
        ("target_enc", target_enc_pipeline, ["brand"]),
    ],
    remainder="drop",
)

poly_pipeline = Pipeline(
    steps=[
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("poly_scaler", StandardScaler()),
    ]
)


def sk(k=11):
    mi_fixed = partial(mutual_info_regression, random_state=42)
    sk_pipeline = Pipeline(steps=[("sk", SelectKBest(score_func=mi_fixed, k=k))])
    return sk_pipeline
