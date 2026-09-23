"""
MLflow logging utilities for parameters, metrics, models, and artifacts.
"""

import json
from pathlib import Path

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec


def log_model_parameters(model_type="RandomForestClassifier", hyperparams=None):
    """Log model hyperparameters to MLflow."""
    if hyperparams is None:
        hyperparams = {}

    mlflow.log_param("model_type", model_type)

    for key, value in hyperparams.items():
        mlflow.log_param(key, value)


def log_model_metrics(metrics_dict):
    """Log model evaluation metrics to MLflow."""
    metrics_to_log = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    for metric in metrics_to_log:
        if metric in metrics_dict:
            mlflow.log_metric(metric, metrics_dict[metric])


def log_artifacts_and_model(model, model_name="late_delivery_classifier"):
    """
    Log the trained model, transformers, and feature list to MLflow.

    Args:
        model: Trained scikit-learn model
        model_name: Name for the model in registry
    """

    project_root = Path(__file__).resolve().parent.parent

    # Define input and output schemas
    # Input: 64 features
    input_schema = Schema([
        ColSpec("double", feature)
        for feature in [
            "n_items",
            "n_distinct_products",
            "n_distinct_sellers",
            "total_price",
            "total_freight",
            "n_payments",
            "total_payment_value",
            "max_installments",
            "estimated_days",
            "purchase_month_sin",
            "purchase_month_cos",
            "purchase_weekday_sin",
            "purchase_weekday_cos",
            "same_state",
            "is_peak_season",

            "customer_state_AC",
            "customer_state_AL",
            "customer_state_AM",
            "customer_state_AP",
            "customer_state_BA",
            "customer_state_CE",
            "customer_state_DF",
            "customer_state_ES",
            "customer_state_GO",
            "customer_state_MA",
            "customer_state_MG",
            "customer_state_MS",
            "customer_state_MT",
            "customer_state_PA",
            "customer_state_PB",
            "customer_state_PE",
            "customer_state_PI",
            "customer_state_PR",
            "customer_state_RJ",
            "customer_state_RN",
            "customer_state_RO",
            "customer_state_RR",
            "customer_state_RS",
            "customer_state_SC",
            "customer_state_SE",
            "customer_state_SP",
            "customer_state_TO",

            "primary_seller_state_AM",
            "primary_seller_state_BA",
            "primary_seller_state_CE",
            "primary_seller_state_DF",
            "primary_seller_state_ES",
            "primary_seller_state_GO",
            "primary_seller_state_MA",
            "primary_seller_state_MG",
            "primary_seller_state_MS",
            "primary_seller_state_MT",
            "primary_seller_state_PA",
            "primary_seller_state_PB",
            "primary_seller_state_PE",
            "primary_seller_state_PI",
            "primary_seller_state_PR",
            "primary_seller_state_RJ",
            "primary_seller_state_RN",
            "primary_seller_state_RO",
            "primary_seller_state_RS",
            "primary_seller_state_SC",
            "primary_seller_state_SE",
            "primary_seller_state_SP",
        ]
    ])

    output_schema = Schema([
        ColSpec("double")
    ])

    signature = ModelSignature(
        inputs=input_schema,
        outputs=output_schema
    )

    # Log the model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name
    )

    # Log transformers as artifacts
    mlflow.log_artifact(
        str(
            project_root
            / "models"
            / "transformers"
            / "numeric_imputer.joblib"
        ),
        artifact_path="transformers"
    )

    mlflow.log_artifact(
        str(
            project_root
            / "models"
            / "transformers"
            / "numeric_scaler.joblib"
        ),
        artifact_path="transformers"
    )

    mlflow.log_artifact(
        str(
            project_root
            / "models"
            / "transformers"
            / "categorical_encoder.joblib"
        ),
        artifact_path="transformers"
    )

    # Log feature list
    mlflow.log_artifact(
        str(
            project_root
            / "data"
            / "feature_list.json"
        ),
        artifact_path="features"
    )


def load_metrics_from_results():
    """Load metrics from models/results_summary.json."""
    project_root = Path(__file__).resolve().parent.parent
    results_file = project_root / "models" / "results_summary.json"

    if results_file.exists():
        with open(results_file, encoding="utf-8") as f:
            results = json.load(f)

        return results

    return {}