"""Lightweight ADC inference runtime for Vercel Functions."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from anarci.anarci import anarci
from Bio.SeqUtils.ProtParam import ProteinAnalysis

ARTIFACT_PATH = Path(__file__).resolve().parent / "model_artifact.json"
ARTIFACT = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
STANDARD_AMINO_ACIDS = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]+")


class SequenceValidationError(ValueError):
    """Raised when an antibody sequence cannot be numbered."""


def clean_sequence(sequence: str) -> str:
    return re.sub(r"\s+", "", str(sequence)).upper()


def validate_sequence(sequence: str, label: str) -> str:
    cleaned = clean_sequence(sequence)
    if not cleaned:
        raise SequenceValidationError(f"{label} 서열을 입력해 주세요.")
    if STANDARD_AMINO_ACIDS.fullmatch(cleaned) is None:
        raise SequenceValidationError(
            f"{label} 서열에는 표준 아미노산 20종의 한 글자 코드만 사용할 수 있습니다."
        )
    return cleaned


def extract_cdr3(sequence: str, expected_chain: str) -> str:
    numbered, alignments, _ = anarci(
        [("query", sequence)],
        scheme="imgt",
        allowed_species=None,
        assign_germline=False,
    )
    if not numbered or numbered[0] is None or not numbered[0]:
        raise SequenceValidationError(
            "서열에서 항체 가변 영역을 찾을 수 없습니다. 전체 가변 영역을 입력해 주세요."
        )

    domains = numbered[0]
    if len(domains) != 1:
        raise SequenceValidationError(
            "한 입력에서 여러 항체 도메인이 발견됐습니다. 단일 chain 서열만 입력해 주세요."
        )
    domain_alignments = alignments[0] or []
    selected_index = None
    for index, alignment in enumerate(domain_alignments):
        chain_type = alignment.get("chain_type")
        if expected_chain == "H" and chain_type == "H":
            selected_index = index
            break
        if expected_chain == "L" and chain_type in {"K", "L"}:
            selected_index = index
            break

    if selected_index is None:
        chain_label = "heavy" if expected_chain == "H" else "light"
        raise SequenceValidationError(
            f"입력한 {chain_label} chain 서열의 항체 도메인을 확인할 수 없습니다."
        )

    positions = domains[selected_index][0]
    cdr3 = "".join(
        amino_acid
        for (position, _), amino_acid in positions
        if 105 <= position < 118 and amino_acid != "-"
    )
    if not cdr3:
        raise SequenceValidationError("서열에서 CDR3 영역을 추출하지 못했습니다.")
    return cdr3


def cdr3_properties(cdr_name: str, sequence: str) -> dict[str, float]:
    protein = ProteinAnalysis(sequence)
    return {
        f"{cdr_name}_pI": protein.isoelectric_point(),
        f"{cdr_name}_net_charge_pH7.4": protein.charge_at_pH(7.4),
        f"{cdr_name}_net_charge_pH6.0": protein.charge_at_pH(6.0),
        f"{cdr_name}_hydrophobicity_GRAVY": protein.gravy(),
    }


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def transformed_features(properties: dict[str, float], subtype: str) -> list[float]:
    numeric_values = []
    for index, feature in enumerate(ARTIFACT["numeric_features"]):
        value = properties.get(feature, ARTIFACT["numeric_imputer"][index])
        numeric_values.append(
            (value - ARTIFACT["numeric_mean"][index])
            / ARTIFACT["numeric_scale"][index]
        )

    subtype_values = [
        1.0 if subtype == known_subtype else 0.0
        for known_subtype in ARTIFACT["model_subtypes"]
    ]
    return numeric_values + subtype_values


def predict_adc_score(
    heavy_sequence: str, light_sequence: str, subtype: str
) -> dict[str, float | str]:
    normalized_subtype = re.sub(r"\s+", " ", str(subtype)).strip()
    if not normalized_subtype:
        raise ValueError("Subtype을 입력해 주세요.")

    heavy = validate_sequence(heavy_sequence, "Heavy chain")
    light = validate_sequence(light_sequence, "Light chain")
    hcdr3 = extract_cdr3(heavy, "H")
    lcdr3 = extract_cdr3(light, "L")

    properties = {
        **cdr3_properties("HCDR3", hcdr3),
        **cdr3_properties("LCDR3", lcdr3),
    }
    features = transformed_features(properties, normalized_subtype)

    status_model = ARTIFACT["status_model"]
    status_score = max(
        1.0,
        min(
            10.0,
            dot_product(status_model["coef"], features)
            + status_model["intercept"],
        ),
    )

    ranking_model = ARTIFACT["ranking_model"]
    ranking_logit = (
        dot_product(ranking_model["coef"], features)
        + ranking_model["intercept"]
    )
    ranking_probability = 1.0 / (1.0 + math.exp(-ranking_logit))
    ranking_score = 1.0 + 9.0 * ranking_probability
    adc_score = max(1.0, min(10.0, 0.90 * status_score + 0.10 * ranking_score))

    return {
        "adc_score": round(adc_score, 3),
        "subtype": normalized_subtype,
        "hcdr3": hcdr3,
        "lcdr3": lcdr3,
    }
