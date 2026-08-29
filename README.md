# Used Car Price Predictor

A web-based **used car price prediction application** built with **Dash** and **Plotly**. Users enter vehicle details through a browser-based form, and an **XGBoost regression model** returns an estimated selling price in USD.

The application supports **partial inputs**. When users leave fields blank, missing values are automatically imputed using statistics derived from the training data or handled internally by the model pipeline.

---

## 📁 Application Structure

```text
car_price_predictor/
├── .Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── code/
    ├── app.py
── saved_models/
        ├── xgb_best_model.pkl
        └── imputation_defaults.json
```

> **Note:** Ensure that the `saved_models/` directory and its model files are available at the expected location before running the application.

---

## 🚀 Quick Start

### Option 1: Run Locally Without Docker

#### Prerequisites

- Python 3.10+
- pip

#### Setup

Navigate to the application directory:

```bash
cd app
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python code/app.py
```

#### Access the Application

Open your browser and navigate to:

```text
http://127.0.0.1:8050
```

---

### Option 2: Run with Docker

Docker is the recommended way to run the application because it provides a consistent environment.

#### Prerequisites

- Docker
- Docker Compose

#### Build and Start

Navigate to the application directory:

```bash
cd app
```

Build the Docker image and start the container:

```bash
docker-compose up --build
```

To run the application in detached mode:

```bash
docker-compose up --build -d
```

#### Access the Application

Open your browser and navigate to:

```text
http://localhost:8050
```
---

## 📝 How to Use

1. Open the application in your web browser.
2. Enter the vehicle details in the prediction form.
3. Provide as much information as you have available.
4. Leave any unknown fields blank if necessary.
5. Click **"Predict Selling Price"**.
6. The application will display the estimated selling price within seconds.

### Supported Input Features

The model expects the following **11 features**:

| Feature | Description |
|---|---|
| `brand` | Vehicle brand |
| `year` | Manufacturing year |
| `km_driven` | Distance driven in kilometers |
| `mileage` | Vehicle mileage |
| `engine` | Engine capacity |
| `max_power` | Maximum engine power |
| `seats` | Number of seats |
| `fuel` | Fuel type |
| `seller_type` | Type of seller |
| `transmission` | Transmission type |
| `owner` | Previous ownership information |

---

## 🔄 Missing Value Handling

The application supports partial user input. Missing values are handled using the following strategies:

### Median Imputation

Numeric fields are filled using their **median values** calculated from the training data:

- `year`
- `km_driven`
- `max_power`

### Mode Imputation

Categorical fields are filled using their **most frequently occurring values (mode)**:

- `brand`
- `fuel`
- `seller_type`
- `transmission`
- `owner`

### Pipeline-Based Imputation

The following fields are handled internally by the model pipeline:

- `mileage`
- `engine`
- `seats`

This allows users to obtain predictions even when some vehicle information is unavailable.

---

## 🐳 Docker Notes

The Docker configuration is set up with the following considerations:

- The Dash application binds to `0.0.0.0:8050` inside the container.
- Container port `8050` is mapped to port `8050` on the host machine.
- The `saved_models/` directory is included in the Docker build.
- The required model files must exist before building the image.

Required model files:

```text
saved_models/
├── xgb_best_model.pkl
└── imputation_defaults.json
```

#### Stop the Container

```bash
docker-compose down
```

---

## 📦 Requirements

The project uses the following key Python dependencies:

```text
dash>=2.14
pandas>=1.5
numpy>=1.24
scikit-learn>=1.3
xgboost>=1.7
joblib>=1.3
```

For the complete list of dependencies and pinned versions, refer to:

```text
requirements.txt
```

---

## 🤖 Machine Learning Model

The application uses an **XGBoost regression model** to estimate the selling price of used vehicles.

The trained model is stored as:

```text
./saved_models/xgb_best_model.pkl
```

Default imputation values are stored separately in:

```text
./saved_models/imputation_defaults.json
```

The model takes vehicle specifications as input and produces an estimated selling price in **Indian Rupees (₹)**.

---

## 🛠️ Technology Stack

- **Python** — Application and machine learning runtime
- **Dash** — Web application framework
- **Plotly** — Data visualization and UI components
- **Pandas** — Data manipulation
- **NumPy** — Numerical computation
- **Scikit-learn** — Machine learning utilities and preprocessing
- **XGBoost** — Regression model
- **Joblib** — Model serialization
- **Docker** — Containerization
- **Docker Compose** — Container orchestration

---

## 📌 Project Overview

The application provides a simple web interface for estimating used-car prices without requiring users to provide every vehicle attribute.

The overall prediction workflow is:

```text
User Input
    │
    ▼
Dash Web Form
    │
    ▼
Input Validation
    │
    ▼
Missing Value Imputation
    │
    ▼
XGBoost Regression Model
    │
    ▼
Estimated Selling Price (₹)
```

