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
 
 
def log_artifacts_and_model(model, feature_columns, model_name="late_delivery_classifier"):
    """
    Log the trained model, transformers, and feature list to MLflow.
 
    Args:
        model: Trained scikit-learn model
        feature_columns: ordered list of the exact column names the model
            expects (e.g. ARTIFACTS.feature_list's numeric + cyclical +
            binary + encoded categorical columns, in that order). Built
            dynamically from the real feature list rather than hardcoded
            here, so the schema can never silently drift out of sync with
            what src/features.py actually produces.
        model_name: Name for the model in registry
    """
 
    project_root = Path(__file__).resolve().parent.parent
 
    input_schema = Schema([ColSpec("double", feature) for feature in feature_columns])
    output_schema = Schema([ColSpec("double")])
 
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
 
    # Log the model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        registered_model_name=model_name,
    )
 
    # Log transformers as artifacts
    for transformer_file in ("numeric_imputer.joblib", "numeric_scaler.joblib", "categorical_encoder.joblib"):
        mlflow.log_artifact(
            str(project_root / "models" / "transformers" / transformer_file),
            artifact_path="transformers",
        )
 
    # Log feature list
    mlflow.log_artifact(
        str(project_root / "data" / "feature_list.json"),
        artifact_path="features",
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