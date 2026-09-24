# Olist Late Delivery Prediction — MLOps Task 3

An end-to-end MLOps pipeline for predicting whether an Olist order will be delivered late.

The project takes a trained machine learning model and turns it into a production-oriented service with data validation, artifact management, experiment/model tracking, testing, API serving, containerization, CI/CD, and monitoring.

---

## 1. Project Overview

The objective of this project is to build a reproducible and maintainable ML service for predicting late delivery of Olist orders.

The prediction target is:

* `is_late = 1` → the order is predicted to be delivered late.
* `is_late = 0` → the order is predicted to be delivered on time.

The prediction is made using information available at the order purchase time.

### Important Data Leakage Rule

The model must not use information that becomes available after the prediction point.

Therefore, post-purchase information such as:

* Actual delivery date
* Carrier delivery date
* Approved date
* Review information

is not used as prediction features.

---

# 2. Project Structure

```text
task3/
│
├── src/
│   ├── api.py
│   ├── artifacts.py
│   ├── features.py
│   ├── logging_config.py
│   ├── mlflow_config.py
│   ├── predict.py
│   └── validation.py
│
├── tests/
│   └── ...
│
├── scripts/
│   └── ...
│
├── notebooks/
│   └── ...
│
├── data/
│   ├── feature_list.json
│   └── sellers_lookup.parquet
│
├── models/
│   └── transformers/
│       ├── numeric_imputer.joblib
│       ├── numeric_scaler.joblib
│       └── categorical_encoder.joblib
│
├── logs/
│   └── app.log
│
├── mlruns/
│   └── ...
│
├── config/
│   └── ...
│
├── docs/
│   └── monitoring.md
│
├── requirements/
│   ├── requirements-dev.txt
│   └── requirements-docker.txt
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .pre-commit-config.yaml
└── README.md
```

---

# 3. Step 1 — Repository Structure and Configuration

The project was organized into separate directories for:

* Source code
* Tests
* Data
* Models
* Configuration
* Notebooks
* Scripts
* Logs
* Documentation
* Requirements

Configuration files were also added to keep project settings and dependencies organized.

The goal of this step was to establish a clean and maintainable project structure before implementing the production pipeline.

---

# 4. Step 2 — Refactor Notebooks into Python Modules

The ML workflow was refactored from notebooks into reusable Python modules.

The main modules include:

### `src/features.py`

Responsible for:

* Validating the structure of incoming orders.
* Building the model feature row.
* Applying the same feature-building logic used during training.

### `src/predict.py`

Provides the main inference entry point:

```python
predict_order(order)
```

It:

1. Validates the input.
2. Builds the feature row.
3. Loads the trained model artifacts.
4. Generates the prediction probability.
5. Applies the selected classification threshold.
6. Returns a structured prediction result.

### `src/artifacts.py`

Loads:

* MLflow model
* Feature list
* Numeric imputer
* Numeric scaler
* Categorical encoder
* Seller lookup

The loaded model is:

```text
late_delivery_classifier
```

with the production alias:

```text
champion
```

and threshold:

```text
0.389
```

---

# 5. Step 3 — Logging and Error Handling

Structured application logging was implemented using the Python logging system.

Logs are stored in:

```text
logs/app.log
```

The prediction pipeline logs:

* Incoming prediction requests
* Validation failures
* Unknown seller warnings
* Prediction results
* Prediction probability
* Prediction latency
* Model name
* Model version
* Unexpected exceptions

Example successful prediction log:

```text
Prediction succeeded:
is_late=1
probability=0.4024
latency=470.0ms
model=late_delivery_classifier
vchampion
```

### Error Handling

The prediction function uses structured error handling so that invalid input does not crash the service.

Errors are returned in the following form:

```json
{
  "ok": false,
  "error": "..."
}
```

Unexpected internal errors are logged with a traceback while a generic error message is returned to the API client.

---

# 6. Step 4 — Data Versioning and Validation

## 6.1 DVC

DVC was used for data versioning and remote artifact storage.

The project uses a Google Drive DVC remote.

The DVC configuration contains a remote for storing versioned data artifacts.

The data pipeline can therefore reproduce the required datasets without storing large data files directly in Git.

The CI pipeline also performs:

```bash
dvc pull
```

before running tests.

## 6.2 Great Expectations

Great Expectations was added for input data validation.

The prediction pipeline validates incoming order data before feature construction.

Validation checks include:

* Required fields
* Data types
* Numeric ranges
* Allowed categorical values
* Non-null requirements

The validation strategy is:

```text
REJECT
```

Invalid data is rejected instead of being silently passed to the model.

---

# 7. Step 5 — MLflow

MLflow was integrated for experiment and model management.

The project uses an MLflow tracking database and MLflow model artifacts.

The configured experiment is:

```text
olist_late_delivery
```

The registered model is:

```text
late_delivery_classifier
```

The production model is accessed using the:

```text
champion
```

alias.

The inference pipeline loads the model through MLflow rather than hard-coding a local model file.

This provides a clear separation between:

* Model training
* Model registration
* Model selection
* Model inference

The model artifact also uses the preprocessing artifacts required to reproduce the training-time feature transformation.

---

# 8. Step 6 — Testing

Automated tests were added using `pytest`.

The test suite covers important parts of the project, including:

* Feature construction
* Input validation
* Prediction behavior
* Error handling
* API functionality

Tests are executed with:

```bash
pytest -v
```

The project also uses Ruff for code quality.

Linting:

```bash
ruff check src scripts tests
```

Formatting:

```bash
ruff format --check src scripts tests
```

Pre-commit hooks are configured to automatically run Ruff and Ruff Formatter.

---

# 9. Step 7 — FastAPI Service

The trained model was exposed through a FastAPI service.

The API provides the following endpoints.

## Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

## Model Information

```http
GET /model
```

Example response:

```json
{
  "model_name": "late_delivery_classifier",
  "model_version": "champion",
  "threshold": 0.389
}
```

## Prediction

```http
POST /predict
```

Example request:

```json
{
  "order_purchase_timestamp": "2018-01-01 10:00:00",
  "order_estimated_delivery_date": "2018-01-15",
  "customer_state": "SP",
  "seller_ids": ["1"],
  "n_items": 1,
  "n_distinct_products": 1,
  "total_price": 100.0,
  "total_freight": 20.0,
  "n_payments": 1,
  "total_payment_value": 120.0,
  "max_installments": 1
}
```

Example response:

```json
{
  "ok": true,
  "is_late": 1,
  "probability": 0.40240362155568443,
  "threshold": 0.389,
  "model_name": "late_delivery_classifier",
  "model_version": "champion",
  "error": null
}
```

The API documentation is automatically available through FastAPI at:

```text
/docs
```

---

# 10. Step 8 — Docker and Docker Compose

The application was containerized using Docker.

## Dockerfile

The Docker image:

* Uses Python 3.10
* Installs production dependencies
* Copies source code
* Copies model artifacts
* Copies data and configuration
* Starts the FastAPI application with Uvicorn

The service listens on:

```text
8000
```

## Docker Compose

Docker Compose defines three services:

### PostgreSQL

```text
postgres
```

Used for the project database.

### FastAPI

```text
api
```

Runs the prediction service.

### Artifact Storage

```text
artifact-storage
```

Uses Nginx to expose MLflow artifact files.

The Compose configuration also includes a PostgreSQL health check so that dependent services wait for the database to become healthy.

---

# 11. Step 9 — CI/CD

GitHub Actions was configured to automate the project checks and Docker image build.

The workflow is located at:

```text
.github/workflows/ci.yml
```

The workflow performs the following operations:

1. Checkout repository.
2. Set up Python 3.10.
3. Install development dependencies.
4. Install Docker dependencies.
5. Install DVC with Google Drive support.
6. Configure DVC credentials through GitHub Secrets.
7. Pull DVC-managed data.
8. Run Ruff linting.
9. Run Ruff formatting checks.
10. Run the test suite.
11. Log in to GitHub Container Registry.
12. Build the Docker image.
13. Push the Docker image to GHCR.

The Docker image is tagged as:

```text
latest
```

DVC credentials are stored securely in GitHub Secrets rather than being committed to the repository.

---

# 12. Step 10 — Monitoring

Monitoring was implemented using Prometheus metrics and application logs.

## 12.1 Request Count

The FastAPI service exposes:

```text
http_requests_total
```

This tracks HTTP requests by:

* Handler
* HTTP method
* Response status

Example:

```text
http_requests_total{handler="/predict",method="POST",status="2xx"}
```

---

## 12.2 Request Latency

The API exposes:

```text
http_request_duration_seconds
```

This metric tracks request processing latency.

It can be used to detect performance degradation and unusually slow API responses.

---

## 12.3 Prediction Distribution

The prediction pipeline exposes:

```text
prediction_total
```

with the label:

```text
is_late
```

The metric distinguishes between:

```text
is_late="0"
```

and:

```text
is_late="1"
```

This allows the distribution of predictions to be monitored over time.

---

## 12.4 Prediction Errors

The API exposes:

```text
prediction_errors_total
```

This counts failed prediction requests.

The prediction error rate can be calculated as:

```text
prediction errors / total prediction requests
```

A sustained increase in the error rate should trigger an investigation.

---

## 12.5 Prediction Logs

Prediction logs are stored in:

```text
logs/app.log
```

The logs contain information such as:

* Prediction result
* Probability
* Latency
* Model name
* Model version
* Validation failures
* Unexpected errors

The logs allow predictions to be evaluated later when the actual delivery date becomes available.

---

## 12.6 Drift Monitoring

The prediction distribution should be monitored over time.

A significant and sustained change in the proportion of:

```text
is_late=0
```

and:

```text
is_late=1
```

may indicate a change in incoming data or model behavior.

Future versions can add statistical input-data drift detection by comparing recent production data with the training distribution.

---

# 13. Alerting Strategy

The following conditions should trigger an alert or investigation.

## High Prediction Error Rate

Alert when the prediction error rate remains above an agreed threshold for a sustained period.

Purpose:

Detect API or model-serving problems.

## High API Latency

Alert when API latency becomes significantly higher than the established baseline.

Purpose:

Detect performance degradation.

## Prediction Distribution Change

Alert when the proportion of late versus on-time predictions changes significantly from the established baseline.

Purpose:

Detect possible changes in data or model behavior.

## API Availability Problem

Alert when the health endpoint fails or the API becomes unavailable.

Purpose:

Detect service outages.

The complete monitoring and alerting documentation is available at:

```text
docs/monitoring.md
```

---

# 14. Running the Project Locally

## Create and activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\activate
```

## Install dependencies

Development dependencies:

```powershell
python -m pip install -r requirements/requirements-dev.txt
```

Production/Docker dependencies:

```powershell
python -m pip install -r requirements/requirements-docker.txt
```

## Run the API

```powershell
python -m uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Metrics:

```text
http://127.0.0.1:8000/metrics
```

---

# 15. Running Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Run Ruff:

```powershell
python -m ruff check src scripts tests
```

Check formatting:

```powershell
python -m ruff format --check src scripts tests
```

Run all pre-commit hooks:

```powershell
python -m pre_commit run --all-files
```

---

# 16. Running with Docker Compose

Build and start the services:

```powershell
docker compose up --build
```

Stop the services:

```powershell
docker compose down
```

The FastAPI service is exposed on:

```text
http://localhost:8000
```

Prometheus metrics are available at:

```text
http://localhost:8000/metrics
```

---

# 17. Model Information

The production inference configuration currently uses:

```text
Model name: late_delivery_classifier
Model version: champion
Threshold: 0.389
Number of features: 64
Known sellers: 3095
```

The classification threshold is intentionally not the default `0.5`; the selected threshold is part of the model inference configuration.

---

# 18. MLOps Pipeline

The final workflow can be summarized as:

```text
Data
  ↓
DVC
  ↓
Data Validation
  ↓
Feature Engineering
  ↓
Trained Model
  ↓
MLflow
  ↓
Artifact Loading
  ↓
Prediction Pipeline
  ↓
FastAPI
  ↓
Docker
  ↓
CI/CD
  ↓
Monitoring + Logs
```

This creates a complete path from versioned data and model artifacts to a deployable and monitored prediction service.

---

# 19. Technologies Used

* Python 3.10
* Pandas
* Scikit-learn
* FastAPI
* Pydantic
* Uvicorn
* Great Expectations
* DVC
* MLflow
* Prometheus
* Docker
* Docker Compose
* PostgreSQL
* pytest
* Ruff
* pre-commit
* Git
* GitHub Actions
* GitHub Container Registry

---

# 20. Task 3 Completion Checklist

| Requirement                            | Status |
| -------------------------------------- | ------ |
| Repository structure and configuration | ✅      |
| Notebook-to-module refactoring         | ✅      |
| Logging and error handling             | ✅      |
| DVC data versioning                    | ✅      |
| Great Expectations validation          | ✅      |
| MLflow integration                     | ✅      |
| Automated testing                      | ✅      |
| FastAPI service                        | ✅      |
| Docker and Docker Compose              | ✅      |
| CI/CD pipeline                         | ✅      |
| Request metrics                        | ✅      |
| Latency metrics                        | ✅      |
| Prediction error monitoring            | ✅      |
| Prediction distribution monitoring     | ✅      |
| Prediction logging                     | ✅      |
| Alerting strategy documented           | ✅      |

---

# 21. Conclusion

Task 3 implements an end-to-end MLOps workflow around the Olist late-delivery prediction model.

The project is designed to make the ML system reproducible, testable, deployable, and observable.

The final system provides:

* Versioned data
* Validated inputs
* Reusable feature engineering
* Managed MLflow model artifacts
* Structured logging
* Automated tests
* REST API
* Docker deployment
* CI/CD automation
* Prometheus monitoring
* Prediction logging
* Alerting documentation
