"""Export the fitted scikit-learn pipelines as a lightweight JSON artifact."""

from __future__ import annotations

import json
from pathlib import Path

from adc_model import ADCScoreModel, FINAL_FEATURE_COLUMNS

OUTPUT_PATH = Path(__file__).resolve().parent / "lib" / "model_artifact.json"


def to_list(values):
    return [float(value) for value in values]


def main() -> None:
    model = ADCScoreModel()
    status_preprocessor = model.status_model.named_steps["preprocessor"]
    numeric_pipeline = status_preprocessor.named_transformers_["numeric"]
    categorical_pipeline = status_preprocessor.named_transformers_["categorical"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]
    encoder = categorical_pipeline.named_steps["onehot"]

    status_regressor = model.status_model.named_steps["model"]
    ranking_classifier = model.ranking_model.named_steps["model"]

    artifact = {
        "version": 1,
        "training_count": model.training_count,
        "numeric_features": FINAL_FEATURE_COLUMNS[1:],
        "numeric_imputer": to_list(imputer.statistics_),
        "numeric_mean": to_list(scaler.mean_),
        "numeric_scale": to_list(scaler.scale_),
        "model_subtypes": [str(value) for value in encoder.categories_[0]],
        "subtypes": model.subtypes,
        "status_model": {
            "coef": to_list(status_regressor.coef_),
            "intercept": float(status_regressor.intercept_),
        },
        "ranking_model": {
            "coef": to_list(ranking_classifier.coef_[0]),
            "intercept": float(ranking_classifier.intercept_[0]),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Exported model artifact: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
