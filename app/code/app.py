import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import joblib
import json,os
from pathlib import Path

# LOAD ASSETS 
BASE_DIR = Path(__file__).resolve().parent
saved_data_path = BASE_DIR.parent / 'saved_models'

# Load defaults for fields NOT handled by the pipeline's ColumnTransformer
# Pipeline handles: mileage, engine (median) | seats (most_frequent)
# App handles: brand, year, km_driven, max_power, fuel, seller_type, transmission, owner
model = joblib.load(saved_data_path / 'xgb_best_model.pkl')
with open(saved_data_path / 'imputation_defaults.json', 'r') as f:
    defaults = json.load(f)


app = dash.Dash(__name__)
app.title = "Car Price Predictor"

app.layout = html.Div(style={'maxWidth': '750px', 'margin': 'auto', 'padding': '20px', 'fontFamily': 'Segoe UI, sans-serif'}, children=[
    
    html.Div([
        html.H1("🚗 Used Car Price Predictor", 
                style={'color': '#1a1a2e', 'marginBottom': '8px'}),
        html.P("Instantly estimate the selling price of a used car using a machine learning model trained on thousands of real vehicle listings.", 
               style={'color': '#4a4a6a', 'fontSize': '16px', 'lineHeight': '1.5'})
    ], style={'textAlign': 'center', 'marginBottom': '30px'}),
    
    html.Div([
        html.H3("How it works", style={'color': '#16213e', 'marginBottom': '12px'}),
        html.P("Enter the vehicle details in the form below. Our XGBoost regression model analyzes the information and returns an estimated selling price in USD.", 
               style={'color': '#333', 'fontSize': '15px', 'lineHeight': '1.6'}),
        
        html.Div([
            html.Strong("📝 Don't have all the details? No problem.", 
                        style={'color': '#0f3460', 'fontSize': '15px'}),
            html.P("If you leave a field blank, the app automatically fills it using imputation techniques learned in class:", 
                   style={'marginTop': '6px', 'color': '#333', 'fontSize': '14px', 'lineHeight': '1.6'}),
            html.Ul([
                html.Li([html.Strong("Numeric fields "), "(Year, KM Driven, Max Power): imputed with the ", html.Em("median"), " from the training data."]),
                html.Li([html.Strong("Categorical fields "), "(Brand, Fuel, Seller Type, Transmission, Owner): imputed with the ", html.Em("mode (most frequent value)"), " from the training data."]),
                html.Li([html.Strong("Pipeline fields "), "(Mileage, Engine, Seats): handled internally by the model pipeline using median / most-frequent imputation."])
            ], style={'color': '#333', 'fontSize': '14px', 'lineHeight': '1.7', 'paddingLeft': '20px'})
        ], style={'backgroundColor': '#f8f9fa', 'padding': '18px', 'borderRadius': '8px', 
                  'borderLeft': '4px solid #0f3460', 'marginTop': '16px'})
    ], style={'marginBottom': '30px'}),
    
    html.Hr(),
    
    html.H3("Vehicle Details", style={'color': '#16213e'}),

    html.Div([
        html.Label("Brand"),
        dcc.Input(id='brand', type='text', placeholder='e.g. Maruti, Hyundai, Toyota, Mahindra', style={'width': '100%'})
    ]),
    html.Br(),
    
    # Numeric inputs
    html.Div([
        html.Div([
            html.Label("Year of Manufacture"),
            dcc.Input(id='year', type='number', placeholder='e.g. 2015', style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Kilometers Driven"),
            dcc.Input(id='km_driven', type='number', placeholder='e.g. 50000', style={'width': '100%'})
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    html.Br(),
    
    html.Div([
        html.Div([
            html.Label("Mileage (kmpl)"),
            dcc.Input(id='mileage', type='number', placeholder='e.g. 18.5', style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Engine (CC)"),
            dcc.Input(id='engine', type='number', placeholder='e.g. 1197', style={'width': '100%'})
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    html.Br(),
    
    html.Div([
        html.Div([
            html.Label("Max Power (bhp)"),
            dcc.Input(id='max_power', type='number', placeholder='e.g. 82', style={'width': '100%'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Seats"),
            dcc.Input(id='seats', type='number', placeholder='e.g. 5', style={'width': '100%'})
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    html.Br(), html.Hr(),
    
    # Categorical inputs
    html.Div([
        html.Div([
            html.Label("Fuel Type"),
            dcc.Dropdown(
                id='fuel',
                options=[
                    {'label': 'Petrol', 'value': 'Petrol'},
                    {'label': 'Diesel', 'value': 'Diesel'}
                ],
                placeholder='Select fuel type'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Seller Type"),
            dcc.Dropdown(
                id='seller_type',
                options=[
                    {'label': 'Individual', 'value': 'Individual'},
                    {'label': 'Dealer', 'value': 'Dealer'},
                    {'label': 'Trustmark Dealer', 'value': 'Trustmark Dealer'}
                ],
                placeholder='Select seller type'
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    html.Br(),
    
    html.Div([
        html.Div([
            html.Label("Transmission"),
            dcc.Dropdown(
                id='transmission',
                options=[
                    {'label': 'Manual', 'value': 'Manual'},
                    {'label': 'Automatic', 'value': 'Automatic'}
                ],
                placeholder='Select transmission'
            )
        ], style={'width': '48%', 'display': 'inline-block'}),
        
        html.Div([
            html.Label("Owner"),
            dcc.Dropdown(
                id='owner',
                options=[
                    {'label': 'First Owner', 'value': 'First Owner'},
                    {'label': 'Second Owner', 'value': 'Second Owner'},
                    {'label': 'Third Owner', 'value': 'Third Owner'},
                    {'label': 'Fourth & Above', 'value': 'Fourth & Above'}
                ],
                placeholder='Select owner type'
            )
        ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
    ]),
    
    html.Br(), html.Hr(),
    
    html.Button('Predict Selling Price', id='predict-btn', n_clicks=0, 
                style={'width': '100%', 'padding': '12px', 'fontSize': '16px', 'cursor': 'pointer'}),
    
    html.Div(id='prediction-output', style={'marginTop': '30px', 'textAlign': 'center'})
])

#  CALLBACK 
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('brand', 'value'),
    State('year', 'value'),
    State('km_driven', 'value'),
    State('mileage', 'value'),
    State('engine', 'value'),
    State('max_power', 'value'),
    State('seats', 'value'),
    State('fuel', 'value'),
    State('seller_type', 'value'),
    State('transmission', 'value'),
    State('owner', 'value')
)
def predict_price(n_clicks, brand, year, km_driven, mileage, engine, max_power, seats,
                  fuel, seller_type, transmission, owner):
    
    if n_clicks == 0:
        return ""
    
    input_data = {
        'brand': [brand],
        'year': [year],
        'km_driven': [km_driven],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner]
    }
    df = pd.DataFrame(input_data)
    
    # APP-LEVEL IMPUTATION
    # These columns have NO imputer inside the ColumnTransformer, so we must fill them here.
    # Pipeline handles: mileage, engine, seats  ->  we leave them as NaN for the pipeline.
    app_impute_cols = ['brand', 'year', 'km_driven', 'max_power', 
                       'fuel', 'seller_type', 'transmission', 'owner']
    
    for col in app_impute_cols:
        val = df[col].iloc[0]
        if pd.isna(val) or val is None or (isinstance(val, str) and val.strip() == ''):
            df[col] = defaults.get(col, np.nan)
    
    # Ensure correct dtypes
    # Numeric columns that the APP imputes
    for col in ['year', 'km_driven', 'max_power']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().iloc[0]:
            df[col] = defaults.get(col, 0)
    
    # Numeric columns that the PIPELINE imputes: convert to numeric but KEEP NaN
    # so the pipeline's SimpleImputer(strategy='median') can fill them.
    for col in ['mileage', 'engine', 'seats']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Predict
    try:
        prediction = np.exp(model.predict(df)[0])
        return html.Div([
            html.H2(f"Predicted Price: ${prediction:,.0f}", style={'color': '#2E8B57'}),
            html.P("Missing fields were automatically filled with typical values from the training data.", 
                   style={'color': 'gray', 'fontSize': '14px'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("Prediction Error", style={'color': '#DC143C'}),
            html.P(str(e), style={'color': 'gray'})
        ])

#  RUN
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)