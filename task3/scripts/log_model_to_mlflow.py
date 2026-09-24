"""
One-time script: register the model that was already trained and saved in
Notebook 6 with MLflow -- parameters, metrics, the model itself, and its
supporting artifacts (transformers, feature list).

This does NOT train anything (training stays in the notebooks). It reads
what Notebook 6 already produced (models/late_delivery_model.joblib and
models/results_summary.json) and logs it as a proper MLflow run, then
registers it in the Model Registry and points the "champion" alias at it
-- the alias the running service will load (see src/artifacts.py).

Run this once (and again whenever a newly trained model should replace the
one currently in production):
    python -m scripts.log_model_to_mlflow
"""

import ast
import json

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src.config import CONFIG, PROJECT_ROOT
from src.mlflow_config import configure_mlflow, REGISTERED_MODEL_NAME, MODEL_ALIAS
from src.mlflow_logging import (
    log_model_parameters,
    log_model_metrics,
    log_artifacts_and_model,
    load_metrics_from_results,
)
from src.logging_config import get_logger

logger = get_logger(__name__)


def _parse_best_config(results: dict) -> dict:
    """
    Notebook 6 stores the winning hyperparameter combination as the
    *string representation* of a dict (it was a pandas index built from
    str(params) during manual grid search). We parse it back into a real
    dict so it can be logged as individual MLflow parameters.
    """
    raw = results.get("best_config")
    if not raw:
        return {}
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        logger.warning(
            "Could not parse best_config %r as a dict; skipping hyperparameters.", raw
        )
        return {}


def main():
    configure_mlflow()

    model_path = PROJECT_ROOT / CONFIG["artifacts"]["model_path"]
    feature_list_path = PROJECT_ROOT / CONFIG["artifacts"]["feature_list_path"]

    model = joblib.load(model_path)
    with open(feature_list_path, "r", encoding="utf-8") as f:
        feature_list = json.load(f)

    # Same column order src/features.py assembles: numeric + cyclical +
    # binary, then the one-hot encoded categorical columns.
    feature_columns = (
        feature_list["numeric_features"]
        + feature_list["cyclical_features"]
        + feature_list["binary_features"]
        + feature_list["categorical_features_encoded"]
    )

    results = load_metrics_from_results()
    hyperparams = _parse_best_config(results)
    final_metrics = results.get("final_test_metrics", {})

    with mlflow.start_run(run_name="register_notebook6_model") as run:
        log_model_parameters(
            model_type="RandomForestClassifier", hyperparams=hyperparams
        )
        log_model_metrics(final_metrics)
        log_artifacts_and_model(
            model, feature_columns, model_name=REGISTERED_MODEL_NAME
        )
        run_id = run.info.run_id

    logger.info("Logged MLflow run %s for model '%s'", run_id, REGISTERED_MODEL_NAME)

    # --- Point the alias the service loads (e.g. "champion") at this version ---
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    # search_model_versions doesn't guarantee order; pick the version tied
    # to the run we just created.
    matching = [v for v in versions if v.run_id == run_id]
    if not matching:
        raise RuntimeError(
            f"Could not find a registered version for run {run_id} -- "
            "registration may have failed silently."
        )
    new_version = matching[0].version

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, MODEL_ALIAS, new_version)
    logger.info(
        "Set alias '%s' -> %s version %s",
        MODEL_ALIAS,
        REGISTERED_MODEL_NAME,
        new_version,
    )

    print(f"Run ID: {run_id}")
    print(f"Registered model: {REGISTERED_MODEL_NAME}, version {new_version}")
    print(f"Alias '{MODEL_ALIAS}' now points to version {new_version}")
    print(f"Load it later with: models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}")


if __name__ == "__main__":
    main()
