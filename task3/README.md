# Olist Late Delivery Prediction — MLOps Pipeline

Predicts whether a newly placed order will be delivered late or on time,
based on order, customer, payment, and seller information available at
checkout time.

This repository turns the notebooks from Task 2 into a real inference
service (Task 3): scripts, tests, an API, and containers.

## Project structure

olist-mlops/
- app/            API service (FastAPI) - built in a later step
- config/         config.yaml - all paths and parameters, no hardcoding
- data/           Parquet datasets and EDA findings from the notebooks
- models/         Trained model + fitted transformers (imputer, scaler, encoder)
- notebooks/      Original Task 2 notebooks (01-06) - kept for reference only
- requirements/   Pinned dependencies (runtime vs. dev)
- src/            Reusable Python modules (config loading, preprocessing, etc.)
- tests/          Unit and integration tests - built in a later step
- README.md

## Setup - running this project from zero

1. Clone the repository

   git clone <repo-url>
   cd olist-mlops

2. Create and activate a virtual environment

   python -m venv venv
   venv\Scripts\activate

3. Install dependencies

   pip install -r requirements/requirements.txt

4. Check the configuration

   All paths, database settings, and the prediction threshold live in
   config/config.yaml. Nothing is hardcoded elsewhere in the code.

   python src/config.py

   This should print the full loaded configuration as JSON.

## Status

- [x] Step 1 - Repository structure & configuration
- [ ] Step 2 - Notebooks refactored into Python modules
- [ ] Step 3 - Logging & error handling
- [ ] Step 4 - Data versioning (DVC) & validation (Great Expectations)
- [ ] Step 5 - Experiment tracking & model registry (MLflow)
- [ ] Step 6 - Testing
- [ ] Step 7 - API (FastAPI)
- [ ] Step 8 - Docker & Docker Compose
- [ ] Step 9 - CI/CD
- [ ] Step 10 - Monitoring