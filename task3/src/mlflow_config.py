"""
MLflow configuration and utilities for experiment tracking and model registry.
"""

import os
from pathlib import Path
from mlflow import set_tracking_uri, set_experiment
import mlflow


# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MLflow tracking directory
MLFLOW_TRACKING_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = f"file:///{MLFLOW_TRACKING_DIR}"

# Experiment names
EXPERIMENT_NAME = "olist_late_delivery"
MODEL_REGISTRY_NAME = "late_delivery_classifier"


def configure_mlflow():
    """Configure MLflow tracking URI and experiment."""
    # Set tracking URI to local directory
    set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    
    # Set active experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    return experiment_id


def get_mlflow_uri():
    """Return the MLflow tracking URI."""
    return MLFLOW_TRACKING_URI


def get_experiment_name():
    """Return the experiment name."""
    return EXPERIMENT_NAME

