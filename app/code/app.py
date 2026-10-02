import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import json
import os
import mlflow.pyfunc
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
local_saved_path = CURRENT_DIR.parent.parent / 'saved_models'
docker_saved_path = CURRENT_DIR.parent / 'saved_models'
SAVED_PATH = local_saved_path if local_saved_path.exists() else docker_saved_path
CLF_PATH = SAVED_PATH / 'classification'

APP_LEVEL_IMPUTE_COLS = ['brand', 'year', 'km_driven', 'max_power',
                         'fuel', 'seller_type', 'transmission', 'owner']
NUMERIC_APP_IMPUTE_COLS = ['year', 'km_driven', 'max_power']
PIPELINE_NUMERIC_COLS = ['mileage', 'engine', 'seats']

with open(SAVED_PATH / 'imputation_defaults.json', 'r') as f:
    DEFAULTS = json.load(f)


def build_input_frame(brand, year, km_driven, mileage, engine, max_power, seats,
                       fuel, seller_type, transmission, owner):
    """Builds a single-row DataFrame and applies the shared app-level
    imputation (median for numeric / mode for categorical) that all
    models rely on for the fields their sklearn pipelines don't impute
    themselves."""
    df = pd.DataFrame([{
        'brand': brand, 'year': year, 'km_driven': km_driven, 'mileage': mileage,
        'engine': engine, 'max_power': max_power, 'seats': seats, 'fuel': fuel,
        'seller_type': seller_type, 'transmission': transmission, 'owner': owner
    }])

    for col in APP_LEVEL_IMPUTE_COLS:
        val = df[col].iloc[0]
        if pd.isna(val) or val is None or (isinstance(val, str) and val.strip() == ''):
            df[col] = DEFAULTS.get(col, np.nan)

    for col in NUMERIC_APP_IMPUTE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().iloc[0]:
            df[col] = DEFAULTS.get(col, 0)

    for col in PIPELINE_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ============================================================
# MODEL 1 — Classic XGBoost Regression
# ============================================================
xgb_model = joblib.load(SAVED_PATH / 'xgb_best_model.pkl')


def predict_old_model(df):
    pred_log = xgb_model.predict(df)[0]
    return float(np.exp(pred_log))


# ============================================================
# MODEL 2 — Custom Polynomial Linear Regression
# ============================================================
POLY_ARTIFACT_FILES = {
    'ct': 'ct.pkl',
    'sk': 'sk.pkl',
    'poly': 'poly.pkl',
}

POLY_MODEL_READY = True
POLY_LOAD_ERROR = None
poly_ct = poly_selector = poly_features = poly_theta = None

try:
    poly_ct = joblib.load(SAVED_PATH / POLY_ARTIFACT_FILES['ct'])
    poly_selector = joblib.load(SAVED_PATH / POLY_ARTIFACT_FILES['sk'])
    poly_features = joblib.load(SAVED_PATH / POLY_ARTIFACT_FILES['poly'])

    with open(SAVED_PATH / 'weights4.json', 'r') as f:
        weights_history = json.load(f)
    latest = weights_history
    bias, coef = latest['bias'], latest['coefficients']
    poly_theta = np.array([bias] + list(coef))
except Exception as e:
    POLY_MODEL_READY = False
    POLY_LOAD_ERROR = str(e)


def _linear_predict(X, theta):
    return X @ theta


def predict_new_model(df):
    if not POLY_MODEL_READY:
        raise RuntimeError(f"New model isn't available yet - {POLY_LOAD_ERROR}")
    ct_out = poly_ct.transform(df)
    selected = poly_selector.transform(ct_out)
    poly_out = poly_features.transform(selected)
    X = np.hstack([np.ones((poly_out.shape[0], 1)), poly_out])
    y_log = _linear_predict(X, poly_theta)[0]
    return float(np.exp(y_log))

# ============================================================
# MODEL 3 — Classification Model (MLflow PyFunc + Pipeline)
# ============================================================
CLF_MODEL_READY = True
CLF_LOAD_ERROR = None
clf_ct = clf_selector = clf_poly = clf_model = None

try:
    # 1. Load transformers (checking both root saved_models/ and classification/)
    ct_file = SAVED_PATH / 'CL_colTransformer_pipeline.pkl'
    if not ct_file.exists():
        ct_file = SAVED_PATH / 'classification' / 'CL_colTransformer_pipeline.pkl'

    sel_file = SAVED_PATH / 'CL_selection_pipeline.pkl'
    if not sel_file.exists():
        sel_file = SAVED_PATH / 'classification' / 'CL_selection_pipeline.pkl'

    poly_file = SAVED_PATH / 'CL_poly_pipeline.pkl'
    if not poly_file.exists():
        poly_file = SAVED_PATH / 'classification' / 'CL_poly_pipeline.pkl'

    clf_ct = joblib.load(ct_file)
    clf_selector = joblib.load(sel_file)
    clf_poly = joblib.load(poly_file)

    # 2. Check for "CL model" (space) first, then fallback to "CL_model" (underscore)
    clf_model_dir = (SAVED_PATH / 'classification' / 'CL model').resolve()
    if not clf_model_dir.exists():
        clf_model_dir = (SAVED_PATH / 'classification' / 'CL_model').resolve()

    if not clf_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at: {clf_model_dir}")

    clf_model = mlflow.pyfunc.load_model(str(clf_model_dir))

except Exception as e:
    CLF_MODEL_READY = False
    CLF_LOAD_ERROR = str(e)

# ============================================================
# LOAD CLASSIFICATION TIERS JSON
# ============================================================
TIERS_FILE = SAVED_PATH / 'classification' / 'classification_tiers.json'
PRICE_TIERS = {}

if TIERS_FILE.exists():
    with open(TIERS_FILE, 'r') as f:
        # JSON keys are stored as strings ("0", "1", ...), convert keys to integers
        PRICE_TIERS = {int(k): v for k, v in json.load(f).items()}


# ============================================================
# CLASSIFICATION PREDICTION FUNCTION
# ============================================================
def predict_clf_model(df):
    if not CLF_MODEL_READY:
        raise RuntimeError(f"Classification model isn't available - {CLF_LOAD_ERROR}")
    
    ct_out = clf_ct.transform(df)
    selected = clf_selector.transform(ct_out)
    poly_out = clf_poly.transform(selected)
    
    # Prepend bias/intercept column of 1s
    bias = np.ones((poly_out.shape[0], 1))
    X_clf = np.hstack([bias, poly_out])
    
    pred = clf_model.predict(X_clf)
    
    if isinstance(pred, (pd.Series, np.ndarray)):
        raw_class = int(pred[0])
    else:
        raw_class = int(pred)
        
    # Check if the class metadata was loaded from JSON
    tier_info = PRICE_TIERS.get(raw_class)
    if tier_info:
        # Supports either {"range_str": "$X - $Y"} or direct {"min": X, "max": Y} formats
        if isinstance(tier_info, dict) and 'range_str' in tier_info:
            range_text = tier_info['range_str']
        elif isinstance(tier_info, dict) and 'min' in tier_info and 'max' in tier_info:
            range_text = f"${tier_info['min']:,.0f} - ${tier_info['max']:,.0f}"
        else:
            range_text = str(tier_info)
            
        return f"Class {raw_class} ({range_text})"
    
    return f"Class {raw_class}"

# ============================================================
# APP / STYLE
# ============================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Car Price & Class Predictor"

COLORS = {
    'navy': '#1a1a2e', 'ink': '#16213e', 'accent': '#0f3460',
    'text': '#333', 'muted': '#4a4a6a', 'green': '#2E8B57', 'red': '#DC143C', 'blue': '#1E90FF'
}

NAV_LINK_STYLE = {
    'padding': '10px 18px', 'textDecoration': 'none', 'color': '#fff',
    'fontWeight': '600', 'fontSize': '15px'
}


def navbar():
    return html.Div([
        dcc.Link('Home', href='/', style=NAV_LINK_STYLE),
        dcc.Link('Classic Model', href='/old-model', style=NAV_LINK_STYLE),
        dcc.Link('New Model', href='/new-model', style=NAV_LINK_STYLE),
        dcc.Link('Classification Model', href='/classification-model', style=NAV_LINK_STYLE),
        dcc.Link('Compare', href='/compare', style=NAV_LINK_STYLE),
    ], style={'backgroundColor': COLORS['navy'], 'display': 'flex',
              'justifyContent': 'center', 'borderRadius': '8px', 'marginBottom': '20px',
              'flexWrap': 'wrap'})


# ---------- shared vehicle-details form ----------
def vehicle_form(prefix):
    return html.Div([
        html.Div([
            html.Label("Brand"),
            dcc.Input(id=f'{prefix}-brand', type='text',
                      placeholder='e.g. Maruti, Hyundai, Toyota, Mahindra',
                      persistence=True, persistence_type='session',
                      style={'width': '100%'})
        ]),
        html.Br(),

        html.Div([
            html.Div([
                html.Label("Year of Manufacture"),
                dcc.Input(id=f'{prefix}-year', type='number', placeholder='e.g. 2015',
                          persistence=True, persistence_type='session',
                          style={'width': '100%'})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Kilometers Driven"),
                dcc.Input(id=f'{prefix}-km_driven', type='number', placeholder='e.g. 50000',
                          persistence=True, persistence_type='session',
                          style={'width': '100%'})
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        html.Br(),

        html.Div([
            html.Div([
                html.Label("Mileage (kmpl)"),
                dcc.Input(id=f'{prefix}-mileage', type='number', placeholder='e.g. 18.5',
                          persistence=True, persistence_type='session',
                          style={'width': '100%'})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Engine (CC)"),
                dcc.Input(id=f'{prefix}-engine', type='number', placeholder='e.g. 1197',
                          persistence=True, persistence_type='session',
                          style={'width': '100%'})
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        html.Br(),

        html.Div([
            html.Div([
                html.Label("Max Power (bhp)"),
                dcc.Input(id=f'{prefix}-max_power', type='number', placeholder='e.g. 82',
                          persistence=True, persistence_type='session',
                          style={'width': '100%'})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Seats"),
                dcc.Input(id=f'{prefix}-seats', type='number', placeholder='e.g. 5',
                          persistence=True, persistence_type='session',
                          style={'width': '100%'})
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        html.Br(), html.Hr(),

        html.Div([
            html.Div([
                html.Label("Fuel Type"),
                dcc.Dropdown(id=f'{prefix}-fuel', options=[
                    {'label': 'Petrol', 'value': 'Petrol'},
                    {'label': 'Diesel', 'value': 'Diesel'}
                ], placeholder='Select fuel type',
                    persistence=True, persistence_type='session')
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Seller Type"),
                dcc.Dropdown(id=f'{prefix}-seller_type', options=[
                    {'label': 'Individual', 'value': 'Individual'},
                    {'label': 'Dealer', 'value': 'Dealer'},
                    {'label': 'Trustmark Dealer', 'value': 'Trustmark Dealer'}
                ], placeholder='Select seller type',
                    persistence=True, persistence_type='session')
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        html.Br(),

        html.Div([
            html.Div([
                html.Label("Transmission"),
                dcc.Dropdown(id=f'{prefix}-transmission', options=[
                    {'label': 'Manual', 'value': 'Manual'},
                    {'label': 'Automatic', 'value': 'Automatic'}
                ], placeholder='Select transmission',
                    persistence=True, persistence_type='session')
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                html.Label("Owner"),
                dcc.Dropdown(id=f'{prefix}-owner', options=[
                    {'label': 'First Owner', 'value': 'First Owner'},
                    {'label': 'Second Owner', 'value': 'Second Owner'},
                    {'label': 'Third Owner', 'value': 'Third Owner'},
                    {'label': 'Fourth & Above', 'value': 'Fourth & Above'}
                ], placeholder='Select owner type',
                    persistence=True, persistence_type='session')
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ]),
        html.Br(), html.Hr(),

        html.Button('Run Prediction', id=f'{prefix}-predict-btn', n_clicks=0,
                    style={'width': '100%', 'padding': '12px', 'fontSize': '16px',
                           'cursor': 'pointer'}),

        html.Div(id=f'{prefix}-prediction-output',
                  style={'marginTop': '30px', 'textAlign': 'center'})
    ])


# ---------- pages ----------
def home_layout():
    return html.Div([
        html.H1("Car Price & Category Predictor",
                style={'color': COLORS['navy'], 'textAlign': 'center', 'marginBottom': '8px'}),
        html.P("Select a model below to evaluate vehicle pricing or classification.",
               style={'color': COLORS['muted'], 'textAlign': 'center', 'fontSize': '16px'}),

        html.Div([
            # Card 1: Classic Model
            html.Div([
                html.H3("Classic Model", style={'marginBottom': '10px'}),
                html.P("Our original XGBoost regression model, trained on thousands of real vehicle listings.",
                       style={'fontSize': '14px', 'minHeight': '60px', 'color': COLORS['muted']}),
                dcc.Link(html.Button("Use Classic Model",
                                      style={'width': '100%', 'padding': '10px', 'cursor': 'pointer'}),
                         href='/old-model')
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                      'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                      'margin': '1.5%', 'boxSizing': 'border-box'}),

            # Card 2: New Model
            html.Div([
                html.H3("New Model", style={'marginBottom': '10px'}),
                html.P("A newly developed polynomial regression model — engineered for non-linear price patterns.",
                       style={'fontSize': '14px', 'minHeight': '60px', 'color': COLORS['muted']}),
                dcc.Link(html.Button("Try New Model",
                                      style={'width': '100%', 'padding': '10px', 'cursor': 'pointer'}),
                         href='/new-model')
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                      'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                      'margin': '1.5%', 'boxSizing': 'border-box'}),

            # Card 3: Classification Model
            html.Div([
                html.H3("Classification Model", style={'marginBottom': '10px'}),
                html.P("A custom Logistic Regression model to classify vehicles into distinct price tiers.",
                       style={'fontSize': '14px', 'minHeight': '60px', 'color': COLORS['muted']}),
                dcc.Link(html.Button("Try Classification Model",
                                      style={'width': '100%', 'padding': '10px', 'cursor': 'pointer'}),
                         href='/classification-model')
            ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                      'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                      'margin': '1.5%', 'boxSizing': 'border-box'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),

        html.Div([
            dcc.Link(html.Button("Compare All Models Side by Side",
                                  style={'padding': '12px 24px', 'fontSize': '16px', 'cursor': 'pointer'}),
                     href='/compare')
        ], style={'textAlign': 'center', 'marginTop': '20px'})
    ])


def old_model_layout():
    return html.Div([
        html.H2("Classic Model — XGBoost Regression", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How it works"),
            html.P("Enter vehicle details below. Unfilled fields are automatically "
                   "imputed using learned median/mode values.", style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['accent']}", 'marginBottom': '20px'}),
        html.Hr(),
        vehicle_form('old')
    ])


def new_model_layout():
    children = [
        html.H2("New Model — Polynomial Regression", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How it works"),
            html.P("Transforms features into polynomial space, selects top terms using mutual "
                   "information, and predicts price using gradient descent regression.",
                   style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['accent']}", 'marginBottom': '20px'}),
    ]

    if not POLY_MODEL_READY:
        children.append(html.Div([
            html.Strong("Model load warning:"),
            html.P(f"{POLY_LOAD_ERROR}", style={'color': 'gray', 'marginTop': '6px'})
        ], style={'backgroundColor': '#fff3cd', 'padding': '14px', 'borderRadius': '8px',
                  'marginBottom': '20px'}))

    children += [html.Hr(), vehicle_form('new')]
    return html.Div(children)


def classification_model_layout():
    children = [
        html.H2("Classification Model — Logistic Regression", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How it works"),
            html.P("Processes inputs through custom column transformers, selection, and polynomial "
                   "pipelines before predicting the target price tier class via MLflow PyFunc wrapper.",
                   style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['blue']}", 'marginBottom': '20px'}),
    ]

    if not CLF_MODEL_READY:
        children.append(html.Div([
            html.Strong("Classification model isn't ready:"),
            html.P(f"Error details: {CLF_LOAD_ERROR}",
                   style={'color': 'gray', 'marginTop': '6px'})
        ], style={'backgroundColor': '#fff3cd', 'padding': '14px', 'borderRadius': '8px',
                  'marginBottom': '20px'}))

    children += [html.Hr(), vehicle_form('clf')]
    return html.Div(children)


def compare_layout():
    return html.Div([
        html.H2("Compare All Models Side by Side", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How this page works"),
            html.P("Enter vehicle details once to view predictions from the XGBoost Regression model, "
                   "Polynomial Regression model, and Classification model simultaneously.",
                   style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['accent']}", 'marginBottom': '20px'}),
        html.Hr(),
        vehicle_form('cmp')
    ])


app.layout = html.Div(style={'maxWidth': '900px', 'margin': 'auto', 'padding': '20px',
                              'fontFamily': 'Segoe UI, sans-serif'}, children=[
    dcc.Location(id='url', refresh=False),
    navbar(),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def render_page(pathname):
    if pathname == '/old-model':
        return old_model_layout()
    if pathname == '/new-model':
        return new_model_layout()
    if pathname == '/classification-model':
        return classification_model_layout()
    if pathname == '/compare':
        return compare_layout()
    return home_layout()


# ---------- compare card renderer ----------
def _result_card(model_name, value, error, is_price=True):
    if error is not None:
        body = [
            html.P("Prediction Error", style={'color': COLORS['red'], 'fontWeight': 'bold'}),
            html.P(str(error), style={'color': 'gray', 'fontSize': '12px'})
        ]
    else:
        formatted_val = f"${value:,.0f}" if is_price else f"Class: {value}"
        val_color = COLORS['green'] if is_price else COLORS['blue']
        body = [html.H2(formatted_val, style={'color': val_color, 'margin': '8px 0', 'fontSize': '22px'})]

    return html.Div([html.H4(model_name, style={'marginBottom': '6px', 'fontSize': '15px'})] + body,
                     style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top',
                            'padding': '16px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                            'margin': '1.5%', 'textAlign': 'center', 'boxSizing': 'border-box'})


@app.callback(
    Output('cmp-prediction-output', 'children'),
    Input('cmp-predict-btn', 'n_clicks'),
    State('cmp-brand', 'value'),
    State('cmp-year', 'value'),
    State('cmp-km_driven', 'value'),
    State('cmp-mileage', 'value'),
    State('cmp-engine', 'value'),
    State('cmp-max_power', 'value'),
    State('cmp-seats', 'value'),
    State('cmp-fuel', 'value'),
    State('cmp-seller_type', 'value'),
    State('cmp-transmission', 'value'),
    State('cmp-owner', 'value'),
    prevent_initial_call=True,
)
def compare_predict(n_clicks, brand, year, km_driven, mileage, engine, max_power,
                     seats, fuel, seller_type, transmission, owner):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    df = build_input_frame(brand, year, km_driven, mileage, engine, max_power,
                            seats, fuel, seller_type, transmission, owner)

    old_price = old_error = new_price = new_error = clf_class = clf_error = None
    try:
        old_price = predict_old_model(df)
    except Exception as e:
        old_error = e

    try:
        new_price = predict_new_model(df)
    except Exception as e:
        new_error = e

    try:
        clf_class = predict_clf_model(df)
    except Exception as e:
        clf_error = e

    cards = html.Div([
        _result_card("Classic Model (XGBoost)", old_price, old_error, is_price=True),
        _result_card("New Model (Poly Regression)", new_price, new_error, is_price=True),
        _result_card("Classification Model", clf_class, clf_error, is_price=False),
    ], style={'textAlign': 'center'})

    extra = []
    if old_error is None and new_error is None:
        diff = new_price - old_price
        pct = (diff / old_price * 100) if old_price else 0
        direction = "higher" if diff > 0 else "lower"
        extra.append(html.P(
            f"The new polynomial model's price estimate is ${abs(diff):,.0f} ({abs(pct):.1f}%) "
            f"{direction} than the classic model's.",
            style={'color': 'gray', 'fontSize': '14px', 'marginTop': '12px', 'textAlign': 'center'}
        ))

    return html.Div([cards] + extra)


# ---------- prediction callbacks ----------
def register_regression_callback(prefix, predict_fn):
    @app.callback(
        Output(f'{prefix}-prediction-output', 'children'),
        Input(f'{prefix}-predict-btn', 'n_clicks'),
        State(f'{prefix}-brand', 'value'),
        State(f'{prefix}-year', 'value'),
        State(f'{prefix}-km_driven', 'value'),
        State(f'{prefix}-mileage', 'value'),
        State(f'{prefix}-engine', 'value'),
        State(f'{prefix}-max_power', 'value'),
        State(f'{prefix}-seats', 'value'),
        State(f'{prefix}-fuel', 'value'),
        State(f'{prefix}-seller_type', 'value'),
        State(f'{prefix}-transmission', 'value'),
        State(f'{prefix}-owner', 'value'),
        prevent_initial_call=True,
    )
    def _predict(n_clicks, brand, year, km_driven, mileage, engine, max_power,
                 seats, fuel, seller_type, transmission, owner):
        if not n_clicks:
            raise dash.exceptions.PreventUpdate

        df = build_input_frame(brand, year, km_driven, mileage, engine, max_power,
                                seats, fuel, seller_type, transmission, owner)
        try:
            price = predict_fn(df)
            return html.Div([
                html.H2(f"Predicted Price: ${price:,.0f}", style={'color': COLORS['green']}),
                html.P("Missing fields were automatically imputed from training defaults.",
                       style={'color': 'gray', 'fontSize': '14px'})
            ])
        except Exception as e:
            return html.Div([
                html.H3("Prediction Error", style={'color': COLORS['red']}),
                html.P(str(e), style={'color': 'gray'})
            ])


@app.callback(
    Output('clf-prediction-output', 'children'),
    Input('clf-predict-btn', 'n_clicks'),
    State('clf-brand', 'value'),
    State('clf-year', 'value'),
    State('clf-km_driven', 'value'),
    State('clf-mileage', 'value'),
    State('clf-engine', 'value'),
    State('clf-max_power', 'value'),
    State('clf-seats', 'value'),
    State('clf-fuel', 'value'),
    State('clf-seller_type', 'value'),
    State('clf-transmission', 'value'),
    State('clf-owner', 'value'),
    prevent_initial_call=True,
)
def predict_clf_callback(n_clicks, brand, year, km_driven, mileage, engine, max_power,
                         seats, fuel, seller_type, transmission, owner):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    df = build_input_frame(brand, year, km_driven, mileage, engine, max_power,
                            seats, fuel, seller_type, transmission, owner)
    try:
        pred_class = predict_clf_model(df)
        return html.Div([
            html.H2(f"Predicted Class: {pred_class}", style={'color': COLORS['blue']}),
            html.P("Missing fields were automatically imputed from training defaults.",
                   style={'color': 'gray', 'fontSize': '14px'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("Prediction Error", style={'color': COLORS['red']}),
            html.P(str(e), style={'color': 'gray'})
        ])


register_regression_callback('old', predict_old_model)
register_regression_callback('new', predict_new_model)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8050,
        debug=False,
        dev_tools_props_check=False
    )