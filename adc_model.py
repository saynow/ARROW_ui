"""Training and inference utilities for the ADC score web application."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from anarci.anarci import anarci
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Use Chain.multiple_domains")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

MODEL_CDR_NAMES = ["HCDR3", "LCDR3"]
MODEL_PROPERTY_SUFFIXES = [
    "pI",
    "net_charge_pH7.4",
    "net_charge_pH6.0",
    "hydrophobicity_GRAVY",
]
FINAL_FEATURE_COLUMNS = ["Subtype"] + [
    f"{cdr}_{suffix}"
    for cdr in MODEL_CDR_NAMES
    for suffix in MODEL_PROPERTY_SUFFIXES
]

STATUS_SCORE_MAP = {
    "approved": 10.0,
    "phase 3": 9.0,
    "phase 3(terminated)": 8.0,
    "phase2": 6.5,
    "phase2(terminated)": 5.0,
    "phase 1": 2.5,
    "phase 1(terminated)": 1.0,
}
MODEL_STATUS_MAP = {
    "phase 2/3": "phase 3",
    "phase 1/2": "phase 1",
}
TRAINING_FILES = ["approved.CSV", "phase3.CSV", "phase2.CSV", "phase1.CSV"]


class SequenceValidationError(ValueError):
    """Raised when an antibody sequence cannot be processed."""


def clean_sequence(sequence: str) -> str:
    return re.sub(r"\s+", "", str(sequence)).upper()


def _validate_sequence(sequence: str, label: str) -> str:
    cleaned = clean_sequence(sequence)
    if not cleaned:
        raise SequenceValidationError(f"{label} 서열을 입력해 주세요.")
    if not re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", cleaned):
        raise SequenceValidationError(
            f"{label} 서열에는 표준 아미노산 20종의 한 글자 코드만 사용할 수 있습니다."
        )
    return cleaned


def calculate_cdr3_features(
    heavy_sequence: str, light_sequence: str
) -> dict[str, float | str | None]:
    heavy = _validate_sequence(heavy_sequence, "Heavy chain")
    light = _validate_sequence(light_sequence, "Light chain")

    try:
        cdr3_sequences = {
            "HCDR3": _extract_cdr3(heavy, "H"),
            "LCDR3": _extract_cdr3(light, "L"),
        }
    except Exception as error:
        raise SequenceValidationError(
            "서열에서 CDR3 영역을 찾을 수 없습니다. 전체 항체 가변 영역을 포함했는지 확인해 주세요."
        ) from error

    result: dict[str, float | str | None] = {
        "sequence_error": None,
        "HCDR3_sequence": cdr3_sequences["HCDR3"],
        "LCDR3_sequence": cdr3_sequences["LCDR3"],
    }
    for cdr_name, cdr_sequence in cdr3_sequences.items():
        protein = ProteinAnalysis(cdr_sequence)
        result.update(
            {
                f"{cdr_name}_pI": protein.isoelectric_point(),
                f"{cdr_name}_net_charge_pH7.4": protein.charge_at_pH(7.4),
                f"{cdr_name}_net_charge_pH6.0": protein.charge_at_pH(6.0),
                f"{cdr_name}_hydrophobicity_GRAVY": protein.gravy(),
            }
        )
    return result


def _extract_cdr3(sequence: str, expected_chain: str) -> str:
    numbered, alignments, _ = anarci(
        [("query", sequence)],
        scheme="imgt",
        allowed_species=None,
        assign_germline=False,
    )
    if not numbered or numbered[0] is None:
        raise ValueError("antibody variable domain not found")

    domains = numbered[0]
    if len(domains) != 1:
        raise ValueError("multiple antibody domains found")
    domain_alignments = alignments[0] or []
    selected_index = next(
        (
            index
            for index, alignment in enumerate(domain_alignments)
            if (
                expected_chain == "H"
                and alignment.get("chain_type") == "H"
            )
            or (
                expected_chain == "L"
                and alignment.get("chain_type") in {"K", "L"}
            )
        ),
        None,
    )
    if selected_index is None:
        raise ValueError("expected antibody chain not found")

    cdr3 = "".join(
        amino_acid
        for (position, _), amino_acid in domains[selected_index][0]
        if 105 <= position < 118 and amino_acid != "-"
    )
    if not cdr3:
        raise ValueError("CDR3 region not found")
    return cdr3


def _read_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding).dropna(how="all")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"지원하지 않는 CSV 인코딩입니다: {path}")


def _add_model_features(data: pd.DataFrame) -> pd.DataFrame:
    calculated_rows = []
    for _, row in data.iterrows():
        try:
            features = calculate_cdr3_features(
                row["Heavy Chain Sequence"],
                row["Light Chain Sequence"],
            )
        except SequenceValidationError as error:
            features = {"sequence_error": str(error)}
        calculated_rows.append(features)

    return pd.concat(
        [data.reset_index(drop=True), pd.DataFrame(calculated_rows)],
        axis=1,
    )


def _normalize_status(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _make_preprocessor() -> ColumnTransformer:
    numeric_features = [
        feature for feature in FINAL_FEATURE_COLUMNS if feature != "Subtype"
    ]
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="Unknown"),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ["Subtype"],
            ),
        ]
    )


class ADCScoreModel:
    """Fits the notebook model once and serves individual predictions."""

    def __init__(self) -> None:
        training_frames = [
            _add_model_features(_read_csv(DATASET_DIR / filename))
            for filename in TRAINING_FILES
        ]
        training_data = pd.concat(training_frames, ignore_index=True)
        training_data["actual_status"] = training_data["ADC status"].map(
            _normalize_status
        )
        training_data["model_status"] = training_data["actual_status"].replace(
            MODEL_STATUS_MAP
        )
        training_data["target_score"] = training_data["model_status"].map(
            STATUS_SCORE_MAP
        )
        training_data = training_data[
            training_data["target_score"].notna()
            & training_data["sequence_error"].isna()
        ].reset_index(drop=True)

        model_input = training_data[FINAL_FEATURE_COLUMNS].copy()
        target_score = training_data["target_score"].astype(float)
        target_ranking = (
            training_data["model_status"].isin(["approved", "phase 3"]).astype(int)
        )

        self.status_model = Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                ("model", Ridge(alpha=0.1)),
            ]
        ).fit(model_input, target_score)
        self.ranking_model = Pipeline(
            [
                ("preprocessor", _make_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        C=0.3,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ).fit(model_input, target_ranking)

        self.training_count = len(training_data)
        self.subtypes = sorted(
            training_data["Subtype"].dropna().astype(str).str.strip().unique().tolist()
        )

    def predict(
        self, heavy_sequence: str, light_sequence: str, subtype: str
    ) -> dict[str, float | str]:
        normalized_subtype = re.sub(r"\s+", " ", str(subtype)).strip()
        if not normalized_subtype:
            raise ValueError("Subtype을 입력해 주세요.")

        features = calculate_cdr3_features(heavy_sequence, light_sequence)
        model_input = pd.DataFrame([{**features, "Subtype": normalized_subtype}])
        for feature in FINAL_FEATURE_COLUMNS:
            if feature not in model_input:
                model_input[feature] = np.nan
        model_input = model_input[FINAL_FEATURE_COLUMNS]

        status_score = np.clip(self.status_model.predict(model_input), 1.0, 10.0)
        approved_phase3_probability = self.ranking_model.predict_proba(model_input)[
            :, 1
        ]
        ranking_score = 1.0 + 9.0 * approved_phase3_probability
        adc_score = np.clip(
            0.90 * status_score + 0.10 * ranking_score,
            1.0,
            10.0,
        )

        return {
            "adc_score": float(np.round(adc_score[0], 3)),
            "subtype": normalized_subtype,
            "hcdr3": str(features["HCDR3_sequence"]),
            "lcdr3": str(features["LCDR3_sequence"]),
        }
