
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# ============================================================
# PATHS
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
local_saved_path = CURRENT_DIR.parent.parent / 'saved_models'
docker_saved_path = CURRENT_DIR.parent / 'saved_models'
SAVED_PATH = local_saved_path if local_saved_path.exists() else docker_saved_path

APP_LEVEL_IMPUTE_COLS = ['brand', 'year', 'km_driven', 'max_power',
                         'fuel', 'seller_type', 'transmission', 'owner']
NUMERIC_APP_IMPUTE_COLS = ['year', 'km_driven', 'max_power']
PIPELINE_NUMERIC_COLS = ['mileage', 'engine', 'seats']

with open(SAVED_PATH / 'imputation_defaults.json', 'r') as f:
    DEFAULTS = json.load(f)


def build_input_frame(brand, year, km_driven, mileage, engine, max_power, seats,
                       fuel, seller_type, transmission, owner):
    """Builds a single-row DataFrame and applies the shared app-level
    imputation (median for numeric / mode for categorical) that both
    models rely on for the fields their sklearn pipelines don't impute
    themselves. mileage / engine / seats are left as NaN on purpose so the
    model pipelines' own imputers can fill them."""
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
# MODEL 1 (existing / "Classic") — XGBoost regression pipeline
# ============================================================
xgb_model = joblib.load(SAVED_PATH / 'xgb_best_model.pkl')


def predict_old_model(df):
    pred_log = xgb_model.predict(df)[0]
    return float(np.exp(pred_log))


# ============================================================
# MODEL 2 (new) — custom polynomial linear regression
# Ported from PolyLinearModelProcessing.py
#
# That script assumes three FITTED transformer objects already exist in
# memory: ct (ColumnTransformer), sk (SelectKBest), poly (PolynomialFeatures).
# For a standalone app they must be persisted with joblib.dump(...) at the
# end of training. Rename the values below if you saved them under
# different filenames.
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
    latest = weights_history  # most recent bias / coefficients snapshot
    bias, coef = latest['bias'], latest['coefficients']
    poly_theta = np.array([bias] + list(coef))
except Exception as e:
    POLY_MODEL_READY = False
    POLY_LOAD_ERROR = str(e)


def _linear_predict(X, theta):
    """Reimplementation of Utils.predict(X, theta) from the training
    notebook: X already has the bias (intercept) column of 1s prepended,
    so this is just a linear combination X @ theta. If your real
    Utils.predict does something more than a plain dot product (e.g. a
    link function), swap the body of this function accordingly."""
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
# APP / STYLE
# ============================================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Car Price Predictor"

COLORS = {
    'navy': '#1a1a2e', 'ink': '#16213e', 'accent': '#0f3460',
    'text': '#333', 'muted': '#4a4a6a', 'green': '#2E8B57', 'red': '#DC143C'
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
        dcc.Link('Compare', href='/compare', style=NAV_LINK_STYLE),
    ], style={'backgroundColor': COLORS['navy'], 'display': 'flex',
              'justifyContent': 'center', 'borderRadius': '8px', 'marginBottom': '20px'})


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

        html.Button('Predict Selling Price', id=f'{prefix}-predict-btn', n_clicks=0,
                    style={'width': '100%', 'padding': '12px', 'fontSize': '16px',
                           'cursor': 'pointer'}),

        html.Div(id=f'{prefix}-prediction-output',
                  style={'marginTop': '30px', 'textAlign': 'center'})
    ])


# ---------- pages ----------
def home_layout():
    return html.Div([
        html.H1("Used Car Price Predictor",
                style={'color': COLORS['navy'], 'textAlign': 'center', 'marginBottom': '8px'}),
        html.P("Choose a model below to estimate the selling price of a used car.",
               style={'color': COLORS['muted'], 'textAlign': 'center', 'fontSize': '16px'}),

        html.Div([
            html.Div([
                html.H3("Classic Model"),
                html.P("Our original XGBoost regression model, trained on thousands of "
                       "real vehicle listings."),
                dcc.Link(html.Button("Use Classic Model",
                                      style={'width': '100%', 'padding': '10px', 'cursor': 'pointer'}),
                         href='/old-model')
            ], style={'width': '46%', 'display': 'inline-block', 'verticalAlign': 'top',
                      'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                      'margin': '2%'}),

            html.Div([
                html.H3("New Model"),
                html.P("A newly developed polynomial regression model — see how it compares."),
                dcc.Link(html.Button("Try New Model",
                                      style={'width': '100%', 'padding': '10px', 'cursor': 'pointer'}),
                         href='/new-model')
            ], style={'width': '46%', 'display': 'inline-block', 'verticalAlign': 'top',
                      'padding': '20px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                      'margin': '2%'}),
        ], style={'textAlign': 'center'}),

        html.Div([
            dcc.Link(html.Button("Compare Both Models Side by Side",
                                  style={'padding': '10px 20px', 'cursor': 'pointer'}),
                     href='/compare')
        ], style={'textAlign': 'center', 'marginTop': '10px'})
    ])


def old_model_layout():
    return html.Div([
        html.H2("Classic Model — XGBoost", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How it works"),
            html.P("Enter the vehicle details below. Any field left blank is automatically "
                   "filled in using the median (numeric) or mode (categorical) value learned "
                   "from the training data; mileage, engine and seats are imputed internally "
                   "by the model pipeline.", style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['accent']}", 'marginBottom': '20px'}),
        html.Hr(),
        vehicle_form('old')
    ])


def new_model_layout():
    children = [
        html.H2("New Model — Polynomial Regression", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How it works, and why it's an improvement"),
            html.P("This model expands the same vehicle features into polynomial terms, "
                   "keeps only the most informative ones using mutual-information feature "
                   "selection, and fits a custom gradient-descent linear regression on top. "
                   "That lets it capture non-linear patterns — for example, how a car's "
                   "value tends to drop faster in its first few years — that a plain "
                   "linear model on the raw features would miss. Fill in the same details "
                   "as the classic model below to compare the estimate.",
                   style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['accent']}", 'marginBottom': '20px'}),
    ]

    if not POLY_MODEL_READY:
        children.append(html.Div([
            html.Strong("This model isn't wired up yet."),
            html.P(f"Missing artifact: {POLY_LOAD_ERROR}",
                   style={'color': 'gray', 'marginTop': '6px'})
        ], style={'backgroundColor': '#fff3cd', 'padding': '14px', 'borderRadius': '8px',
                  'marginBottom': '20px'}))

    children += [html.Hr(), vehicle_form('new')]
    return html.Div(children)


def compare_layout():
    return html.Div([
        html.H2("Compare Both Models", style={'color': COLORS['ink']}),
        html.Div([
            html.Strong("How this page works"),
            html.P("Enter the vehicle details once and both models will run on the exact "
                   "same input, so you can compare their estimates side by side.",
                   style={'marginTop': '6px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px',
                  'borderLeft': f"4px solid {COLORS['accent']}", 'marginBottom': '20px'}),
        html.Hr(),
        vehicle_form('cmp')
    ])


app.layout = html.Div(style={'maxWidth': '750px', 'margin': 'auto', 'padding': '20px',
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
    if pathname == '/compare':
        return compare_layout()
    return home_layout()


# ---------- compare callback: runs BOTH models on the same input ----------
def _result_card(model_name, price, error):
    if error is not None:
        body = [
            html.P("Prediction Error", style={'color': COLORS['red'], 'fontWeight': 'bold'}),
            html.P(str(error), style={'color': 'gray', 'fontSize': '13px'})
        ]
    else:
        body = [html.H2(f"${price:,.0f}", style={'color': COLORS['green'], 'margin': '8px 0'})]

    return html.Div([html.H4(model_name, style={'marginBottom': '4px'})] + body,
                     style={'width': '46%', 'display': 'inline-block', 'verticalAlign': 'top',
                            'padding': '16px', 'border': '1px solid #ddd', 'borderRadius': '10px',
                            'margin': '2%', 'textAlign': 'center'})


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

    old_price = old_error = new_price = new_error = None
    try:
        old_price = predict_old_model(df)
    except Exception as e:
        old_error = e
    try:
        new_price = predict_new_model(df)
    except Exception as e:
        new_error = e

    cards = html.Div([
        _result_card("Classic Model (XGBoost)", old_price, old_error),
        _result_card("New Model (Polynomial Regression)", new_price, new_error),
    ], style={'textAlign': 'center'})

    extra = []
    if old_error is None and new_error is None:
        diff = new_price - old_price
        pct = (diff / old_price * 100) if old_price else 0
        direction = "higher" if diff > 0 else "lower"
        extra.append(html.P(
            f"The new model's estimate is ${abs(diff):,.0f} ({abs(pct):.1f}%) "
            f"{direction} than the classic model's.",
            style={'color': 'gray', 'fontSize': '14px', 'marginTop': '10px', 'textAlign': 'center'}
        ))

    return html.Div([cards] + extra)


# ---------- prediction callbacks (one per model, sharing the same logic) ----------
def register_predict_callback(prefix, predict_fn):
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
                html.P("Missing fields were automatically filled with typical values "
                       "from the training data.", style={'color': 'gray', 'fontSize': '14px'})
            ])
        except Exception as e:
            return html.Div([
                html.H3("Prediction Error", style={'color': COLORS['red']}),
                html.P(str(e), style={'color': 'gray'})
            ])

    return _predict


register_predict_callback('old', predict_old_model)
register_predict_callback('new', predict_new_model)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=8050,
        debug=False,
        dev_tools_props_check=False
    )