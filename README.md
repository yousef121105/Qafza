# Qafza
Qafza_Train_ALL_Tasks

# Qafza — MLOps Training Tasks

A complete MLOps training repository developed as part of the Qafza training program.

The repository covers the development of a machine learning system from data preparation and feature engineering to model training, experiment tracking, API deployment, containerization, CI/CD, and monitoring.

The main project uses the **Olist Brazilian E-Commerce dataset** to build a machine learning system that predicts whether an order is likely to be delivered late.

---

## 📌 Project Overview

The main objective of this repository is to apply MLOps practices to a real-world machine learning problem.

The project follows an end-to-end workflow:

```text
Raw Data
   ↓
Data Preparation
   ↓
Database / Data Processing
   ↓
Feature Engineering
   ↓
Data Validation
   ↓
Train / Validation / Test Split
   ↓
Model Training
   ↓
MLflow Experiment Tracking
   ↓
Model Artifacts
   ↓
Prediction API
   ↓
Docker
   ↓
CI/CD
   ↓
Monitoring & Logging
```

The project focuses on predicting:

> **Will an Olist order be delivered late?**

The target variable is:

```text
is_late
```

where:

```text
is_late = 1
```

means the order was delivered later than the estimated delivery date, while:

```text
is_late = 0
```

means it was not delivered late.

---

# 📂 Repository Structure

```text
Qafza/
│
├── task2/
│   └── ...
│
├── task3/
│   ├── src/
│   ├── scripts/
│   ├── tests/
│   ├── models/
│   ├── data/
│   ├── config/
│   ├── notebooks/
│   ├── requirements/
│   ├── docs/
│   ├── artifacts/
│   ├── logs/
│   ├── mlruns/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── pyproject.toml
│   ├── .pre-commit-config.yaml
│   ├── .dvc/
│   └── README.md
│
├── .gitignore
└── README.md
```

The repository is organized so that each training task is maintained separately while remaining part of the same Git repository.

---

# 🧩 Tasks

## Task 1 — Data and Database Preparation

The first stage focuses on preparing the Olist dataset and setting up the data infrastructure required for the following stages.

The Olist dataset consists of multiple CSV tables representing different aspects of the e-commerce platform, including:

* Customers
* Sellers
* Products
* Orders
* Order Items
* Payments
* Reviews
* Product translations
* Geolocation data

The data was prepared and loaded into a PostgreSQL database.

### Main technologies

* Python
* Pandas
* PostgreSQL
* SQLAlchemy
* pgAdmin

### Main goal

Prepare a reliable structured data source that can be used for feature engineering and machine learning.

---

# 🧩 Task 2 — Data Processing and Feature Engineering

Task 2 builds on the prepared Olist data and creates an order-level machine learning dataset.

The different Olist tables are combined to create meaningful features for each order.

The resulting dataset is referred to as:

```text
ml_table
```

### Feature groups

The feature engineering process includes information related to:

* Order information
* Customer information
* Seller information
* Products
* Order items
* Payments
* Geolocation
* Aggregated order-level statistics

Examples include:

```text
n_items
n_distinct_products
total_price
total_freight
n_payments
total_payment_value
max_installments
customer_state
seller information
geolocation features
```

The resulting dataset provides the foundation for the machine learning pipeline implemented in Task 3.

---

# 🧩 Task 3 — End-to-End MLOps Pipeline

Task 3 extends the project into a complete MLOps workflow.

It covers:

1. Repository and project configuration
2. Refactoring notebooks into Python modules
3. Logging and error handling
4. Data versioning and validation
5. MLflow experiment tracking
6. Automated testing
7. FastAPI prediction service
8. Docker and Docker Compose
9. CI/CD
10. Monitoring

---

## 1. Project Configuration

The project was organized into separate directories for source code, data, models, configuration, tests, notebooks, scripts, requirements, documentation, and artifacts.

Configuration files include:

```text
pyproject.toml
.pre-commit-config.yaml
.dvc/config
```

The project uses Python 3.10.

---

# 2. Refactoring Notebooks into Python Modules

The machine learning workflow was refactored from notebooks into reusable Python modules.

This makes the project easier to:

* Test
* Maintain
* Reuse
* Deploy
* Automate

The main source code is located inside:

```text
task3/src/
```

---

# 3. Logging and Error Handling

Structured logging was added to the machine learning pipeline and API.

Logs are stored under:

```text
logs/
```

The logs can contain information such as:

* Input information
* Prediction results
* Prediction probability
* Model information
* Validation warnings
* Validation errors
* Prediction latency
* Runtime errors

The system also handles prediction failures and exposes prediction error metrics.

---

# 4. Data Versioning and Validation

## DVC

**DVC (Data Version Control)** is used for data and artifact versioning.

The project uses DVC to track large data files and machine learning artifacts without storing all of them directly in Git.

The repository is configured with a DVC remote.

The main purpose is to make it possible to reproduce the required project state and retrieve versioned data.

---

## Great Expectations

**Great Expectations** is used to validate the data before it is used by the machine learning pipeline.

Validation helps detect problems such as:

* Missing values
* Incorrect data types
* Unexpected values
* Invalid schema
* Invalid feature data

This provides an additional layer of reliability before prediction.

---

# 5. MLflow

**MLflow** is used for machine learning experiment tracking and model management.

The project tracks:

* Experiments
* Parameters
* Metrics
* Model artifacts
* Model versions

The trained model is registered as:

```text
late_delivery_classifier
```

Current model information:

```text
Model name: late_delivery_classifier
Model version: 1.0.0
Prediction threshold: 0.389
Number of features: 64
Known sellers: 3095
```

The production model uses the:

```text
champion
```

alias.

---

# 🎯 Machine Learning Dataset

The final order-level dataset contains:

```text
99,441 orders
```

The target variable is:

```text
is_late
```

The labeled data contains:

| Class          |  Count | Percentage |
| -------------- | -----: | ---------: |
| Not Late (`0`) | 88,644 |     91.89% |
| Late (`1`)     |  7,826 |      8.11% |

Eight rows without the required delivered-date information were excluded from the labeled dataset.

---

# ⚠️ Data Leakage Prevention

A major requirement of the project is preventing **data leakage**.

The prediction point is:

```text
order_purchase_timestamp
```

Only information available at or before the prediction point should be used as model features.

The following types of information must not be used as prediction features:

* Delivered date
* Carrier delivery date
* Approved date when it would not be available at prediction time
* Customer reviews
* Any information generated after the order prediction point

This ensures that the model represents a realistic production prediction scenario.

---

# 📊 Dataset Split

The data is split according to the order purchase timestamp rather than randomly.

The split is:

```text
70% Training
15% Validation
15% Test
```

The training set contains:

```text
67,529 orders
```

The time-based split helps simulate a real production environment where the model predicts future orders using historical data.

---

# 6. Automated Testing

Automated tests were added using:

```text
pytest
```

Tests cover important parts of the project, including:

* Feature processing
* Prediction behavior
* API behavior
* Payload validation
* Model-related functionality

The project also uses:

```text
Ruff
```

for linting and formatting.

Pre-commit hooks are configured to automatically run code-quality checks.

Example:

```powershell
python -m pre_commit run --all-files
```

---

# 7. FastAPI Prediction Service

A REST API was developed using:

```text
FastAPI
```

The API provides the machine learning model as a prediction service.

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok"
}
```

---

### Model Information

```http
GET /model
```

Returns information about the currently loaded model.

Example:

```json
{
  "model_name": "late_delivery_classifier",
  "model_version": "champion",
  "threshold": 0.389
}
```

---

### Prediction

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

---

# 8. Docker

The application is containerized using:

```text
Docker
Docker Compose
```

The project includes:

```text
Dockerfile
docker-compose.yml
.dockerignore
```

The Docker setup contains services for:

```text
PostgreSQL
FastAPI
Artifact Storage
```

The API runs on:

```text
http://localhost:8000
```

FastAPI documentation is available through:

```text
http://localhost:8000/docs
```

The artifact-storage service is exposed on:

```text
http://localhost:8080
```

---

# 9. CI/CD

GitHub Actions is used to automate the project workflow.

The CI/CD pipeline performs tasks including:

```text
Checkout repository
        ↓
Setup Python
        ↓
Install dependencies
        ↓
Configure DVC
        ↓
Pull versioned data
        ↓
Run Ruff
        ↓
Run formatting checks
        ↓
Run tests
        ↓
Login to GitHub Container Registry
        ↓
Build Docker image
        ↓
Push Docker image
```

The workflow is located at:

```text
.github/workflows/ci.yml
```

The Docker image is published to:

```text
GitHub Container Registry (GHCR)
```

---

# 10. Monitoring

Monitoring was added to the FastAPI service using:

```text
Prometheus
prometheus-fastapi-instrumentator
```

The service exposes metrics through:

```http
GET /metrics
```

## Monitored Metrics

### Request Count

Tracks the number of API requests.

Example:

```text
http_requests_total
```

---

### Request Latency

Tracks AP
