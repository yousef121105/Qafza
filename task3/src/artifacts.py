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


class Artifacts:
    """
    Holds every object the prediction pipeline needs, loaded once at
    import time. One instance of this class (ARTIFACTS, below) is shared
    by the whole application -- nobody should load these files a second time.
    """

    def __init__(self):
        artifacts_cfg = CONFIG["artifacts"]

        # --- The trained model (Notebook 6) ---
        self.model = joblib.load(PROJECT_ROOT / artifacts_cfg["model_path"])

        # --- The fitted transformers (Notebook 5) ---
        # Each of these was fit on the TRAINING split only, and is loaded
        # here as-is. They are applied (transform), never re-fit.
        self.numeric_imputer = joblib.load(
            PROJECT_ROOT / artifacts_cfg["numeric_imputer_path"]
        )
        self.numeric_scaler = joblib.load(
            PROJECT_ROOT / artifacts_cfg["numeric_scaler_path"]
        )
        self.categorical_encoder = joblib.load(
            PROJECT_ROOT / artifacts_cfg["categorical_encoder_path"]
        )

        # --- The feature list (Notebook 5) ---
        # Tells us exactly which columns the model expects, and in what
        # grouping (numeric / cyclical / binary / categorical). The
        # feature-building code below must produce exactly these columns.
        with open(PROJECT_ROOT / artifacts_cfg["feature_list_path"], "r", encoding="utf-8") as f:
            self.feature_list = json.load(f)

        # --- Static seller lookup (built once, see scripts/build_seller_lookup.py) ---
        # Replaces the live database query used in the notebooks: the
        # inference service must not depend on a database connection to
        # answer a single prediction request.
        seller_lookup_path = PROJECT_ROOT / "data" / "sellers_lookup.parquet"
        self.seller_lookup = pd.read_parquet(seller_lookup_path)

        # --- Prediction settings (Notebook 6) ---
        self.threshold = float(CONFIG["prediction"]["threshold"])
        self.model_name = CONFIG["model"]["name"]
        self.model_version = CONFIG["model"]["version"]

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
