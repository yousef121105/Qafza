"""
Load every fitted object the inference pipeline needs, exactly once, and
expose them as a single ARTIFACTS object the rest of the service can import.
 
This module NEVER fits, trains, or refits anything.
 
As of Step 5, the trained model itself is loaded from the MLflow Model
Registry (by alias, e.g. "champion") rather than from a local .joblib file
-- this is what lets the service pick up a newly promoted model without
any code change, and matches the requirement that "the service loads the
model from the registry ... not from a local notebook folder".
 
The transformers (imputer, scaler, encoder) and the feature list are still
read from local files under models/ and data/. They were logged to MLflow
as run artifacts too (see scripts/log_model_to_mlflow.py); loading them
from the registry's artifact store as well is a natural next step, kept
local for now to keep this change focused on the model itself.
"""
 
import json
from pathlib import Path
 
import joblib
import mlflow
import pandas as pd
 
from src.config import CONFIG, PROJECT_ROOT
from src.mlflow_config import configure_mlflow, REGISTERED_MODEL_NAME, MODEL_ALIAS
from src.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
class ArtifactLoadError(RuntimeError):
    """Raised when a required artifact (model, transformer, or file) is
    missing or unreadable.
 
    This is intentionally a distinct exception type (not a bare
    FileNotFoundError) so calling code -- and later, the API startup
    logic in Step 7 -- can catch it specifically and fail with a clear,
    actionable message instead of a raw stack trace.
    """
 
 
def _load_file(label: str, path: Path, loader):
    """Load one local artifact file, logging success or a clear failure reason."""
    if not path.exists():
        message = f"Missing artifact '{label}': expected file at {path}"
        logger.error(message)
        raise ArtifactLoadError(message)
    try:
        obj = loader(path)
    except Exception as e:  # noqa: BLE001 -- we deliberately catch broadly here
        message = f"Failed to load artifact '{label}' from {path}: {e}"
        logger.error(message)
        raise ArtifactLoadError(message) from e
 
    logger.info("Loaded artifact '%s' from %s", label, path)
    return obj
 
 
def _load_model_from_registry():
    """
    Load the model registered under REGISTERED_MODEL_NAME at MODEL_ALIAS.

    MLflow Registry is used to identify the promoted model. The actual model
    is then loaded from its local artifact source path.
    """
    configure_mlflow()

    try:
        client = mlflow.MlflowClient()

        model_version = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
        )

        source = model_version.source

        # MLflow stores the source as a file URI.
        # Inside Docker, the artifact store is mounted/copied under /app/mlruns.
        if source.startswith("file:///app/mlruns/"):
            model_path = source.replace("file://", "")
        else:
            model_path = source

        logger.info(
            "Loading model '%s' @ '%s' from source: %s",
            REGISTERED_MODEL_NAME,
            MODEL_ALIAS,
            model_path,
        )

        model = mlflow.sklearn.load_model(model_path)

    except Exception as e:  # noqa: BLE001
        message = (
            f"Failed to load model '{REGISTERED_MODEL_NAME}' at alias "
            f"'{MODEL_ALIAS}' from the MLflow registry. "
            f"Original error: {e}"
        )
        logger.error(message)
        raise ArtifactLoadError(message) from e

    logger.info(
        "Loaded model from MLflow registry: %s @ %s",
        REGISTERED_MODEL_NAME,
        MODEL_ALIAS,
    )
    return model
 
 
class Artifacts:
    """
    Holds every object the prediction pipeline needs, loaded once at
    import time. One instance of this class (ARTIFACTS, below) is shared
    by the whole application -- nobody should load these files a second time.
    """
 
    def __init__(self):
        artifacts_cfg = CONFIG["artifacts"]
        logger.info("Loading artifacts...")
 
        # --- The trained model, from the MLflow Model Registry ---
        self.model = _load_model_from_registry()
 
        # --- The fitted transformers (Notebook 5) ---
        # Each of these was fit on the TRAINING split only, and is loaded
        # here as-is. They are applied (transform), never re-fit.
        self.numeric_imputer = _load_file(
            "numeric_imputer",
            PROJECT_ROOT / artifacts_cfg["numeric_imputer_path"],
            joblib.load,
        )
        self.numeric_scaler = _load_file(
            "numeric_scaler",
            PROJECT_ROOT / artifacts_cfg["numeric_scaler_path"],
            joblib.load,
        )
        self.categorical_encoder = _load_file(
            "categorical_encoder",
            PROJECT_ROOT / artifacts_cfg["categorical_encoder_path"],
            joblib.load,
        )
 
        # --- The feature list (Notebook 5) ---
        self.feature_list = _load_file(
            "feature_list",
            PROJECT_ROOT / artifacts_cfg["feature_list_path"],
            lambda p: json.loads(p.read_text(encoding="utf-8")),
        )
 
        # --- Static seller lookup (built once, see scripts/build_seller_lookup.py) ---
        self.seller_lookup = _load_file(
            "seller_lookup",
            PROJECT_ROOT / "data" / "sellers_lookup.parquet",
            pd.read_parquet,
        )
 
        # --- Prediction settings (Notebook 6) ---
        self.threshold = float(CONFIG["prediction"]["threshold"])
        self.model_name = REGISTERED_MODEL_NAME
        self.model_version = MODEL_ALIAS
 
        logger.info(
            "All artifacts loaded successfully (model=%s @%s, threshold=%.3f)",
            self.model_name, self.model_version, self.threshold,
        )
 
    def __repr__(self) -> str:
        return (
            f"Artifacts(model={self.model_name} @{self.model_version}, "
            f"threshold={self.threshold}, "
            f"n_features={len(self.feature_list['numeric_features']) + len(self.feature_list['cyclical_features']) + len(self.feature_list['binary_features']) + len(self.feature_list['categorical_features_encoded'])}, "
            f"n_sellers_known={len(self.seller_lookup)})"
        )
 
 
# Loaded once, when this module is first imported anywhere in the app.
# Every other module should import ARTIFACTS from here -- never re-load
# the joblib files, and never re-load the model from the registry, again.
ARTIFACTS = Artifacts()
 
 
if __name__ == "__main__":
    # Manual check: "python -m src.artifacts" from the project root (task3/)
    print(ARTIFACTS)