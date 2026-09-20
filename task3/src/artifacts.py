"""
Load every fitted object produced by the notebooks (Task 2) exactly once,
and expose them as a single ARTIFACTS object the rest of the service can import.
 
This module NEVER fits, trains, or refits anything. It only loads what was
already saved to disk by Notebook 5 (transformers) and Notebook 6 (model).
That is the core rule of the inference pipeline: fit happens in notebooks,
loading happens here.
"""
 
import json
from pathlib import Path
 
import joblib
import pandas as pd
 
from src.config import CONFIG, PROJECT_ROOT
from src.logging_config import get_logger
 
logger = get_logger(__name__)
 
 
class ArtifactLoadError(RuntimeError):
    """Raised when a required artifact file is missing or unreadable.
 
    This is intentionally a distinct exception type (not a bare
    FileNotFoundError) so calling code -- and later, the API startup
    logic in Step 7 -- can catch it specifically and fail with a clear,
    actionable message instead of a raw stack trace.
    """
 
 
def _load_file(label: str, path: Path, loader):
    """Load one artifact file, logging success or a clear failure reason."""
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
 
 
class Artifacts:
    """
    Holds every object the prediction pipeline needs, loaded once at
    import time. One instance of this class (ARTIFACTS, below) is shared
    by the whole application -- nobody should load these files a second time.
    """
 
    def __init__(self):
        artifacts_cfg = CONFIG["artifacts"]
        logger.info("Loading artifacts...")
 
        # --- The trained model (Notebook 6) ---
        self.model = _load_file(
            "model", PROJECT_ROOT / artifacts_cfg["model_path"], joblib.load
        )
 
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
        # Tells us exactly which columns the model expects, and in what
        # grouping (numeric / cyclical / binary / categorical). The
        # feature-building code below must produce exactly these columns.
        self.feature_list = _load_file(
            "feature_list",
            PROJECT_ROOT / artifacts_cfg["feature_list_path"],
            lambda p: json.loads(p.read_text(encoding="utf-8")),
        )
 
        # --- Static seller lookup (built once, see scripts/build_seller_lookup.py) ---
        # Replaces the live database query used in the notebooks: the
        # inference service must not depend on a database connection to
        # answer a single prediction request.
        self.seller_lookup = _load_file(
            "seller_lookup",
            PROJECT_ROOT / "data" / "sellers_lookup.parquet",
            pd.read_parquet,
        )
 
        # --- Prediction settings (Notebook 6) ---
        self.threshold = float(CONFIG["prediction"]["threshold"])
        self.model_name = CONFIG["model"]["name"]
        self.model_version = CONFIG["model"]["version"]
 
        logger.info(
            "All artifacts loaded successfully (model=%s v%s, threshold=%.3f)",
            self.model_name, self.model_version, self.threshold,
        )
 
    def __repr__(self) -> str:
        return (
            f"Artifacts(model={self.model_name} v{self.model_version}, "
            f"threshold={self.threshold}, "
            f"n_features={len(self.feature_list['numeric_features']) + len(self.feature_list['cyclical_features']) + len(self.feature_list['binary_features']) + len(self.feature_list['categorical_features_encoded'])}, "
            f"n_sellers_known={len(self.seller_lookup)})"
        )
 
 
# Loaded once, when this module is first imported anywhere in the app.
# Every other module should import ARTIFACTS from here -- never re-load
# the joblib files themselves.
ARTIFACTS = Artifacts()
 
 
if __name__ == "__main__":
    # Manual check: "python -m src.artifacts" from the project root (task3/)
    print(ARTIFACTS)
 


























