"""
Central MLflow setup: where runs/models are tracked, which experiment they
belong to, and which registered-model alias the service should load at
inference time.
 
Every other module (the one-time registration script, and later
src/artifacts.py) should import from here instead of hardcoding a tracking
URI or experiment name -- same "one source of truth" rule as config.yaml.
"""
 
import mlflow
 
from src.config import CONFIG, PROJECT_ROOT
from src.logging_config import get_logger
 
logger = get_logger(__name__)
 
# All MLflow settings live under the "mlflow:" key in config.yaml, e.g.:
#
# mlflow:
#   tracking_uri: "sqlite:///mlflow.db"
#   experiment_name: "late_delivery_prediction"
#   registered_model_name: "late_delivery_classifier"
#   model_alias: "champion"
_mlflow_cfg = CONFIG["mlflow"]
 
TRACKING_URI = _mlflow_cfg["tracking_uri"]
EXPERIMENT_NAME = _mlflow_cfg["experiment_name"]
REGISTERED_MODEL_NAME = _mlflow_cfg["registered_model_name"]
MODEL_ALIAS = _mlflow_cfg["model_alias"]
 
 
def configure_mlflow() -> None:
    """
    Point MLflow at the right tracking store and experiment. Call this
    once, before starting a run or loading a registered model.
 
    A relative sqlite URI (the default) resolves relative to the current
    working directory, not the project root -- so we rewrite it to an
    absolute path here to make this work no matter where a script is
    launched from.
    """
    tracking_uri = TRACKING_URI
    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        relative_path = tracking_uri.removeprefix("sqlite:///")
        absolute_path = PROJECT_ROOT / relative_path
        tracking_uri = f"sqlite:///{absolute_path.as_posix()}"
 
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        "MLflow configured: tracking_uri=%s experiment=%s",
        tracking_uri, EXPERIMENT_NAME,
    )
 
 
if __name__ == "__main__":
    # Manual check: "python -m src.mlflow_config" from the project root (task3/)
    configure_mlflow()
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment:   {EXPERIMENT_NAME}")
    print(f"Registered model name: {REGISTERED_MODEL_NAME}")
    print(f"Model alias to load:   {MODEL_ALIAS}")
 