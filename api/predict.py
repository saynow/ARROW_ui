"""Vercel Function for ADC score metadata and prediction."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler

from lib.adc_runtime import (
    ARTIFACT,
    SequenceValidationError,
    predict_adc_score,
)


class handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self.send_json(
            {
                "subtypes": ARTIFACT["subtypes"],
                "training_count": ARTIFACT["training_count"],
            }
        )

    def do_POST(self) -> None:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 200_000:
                raise ValueError("요청 데이터의 크기가 올바르지 않습니다.")

            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = predict_adc_score(
                heavy_sequence=payload.get("heavy_sequence", ""),
                light_sequence=payload.get("light_sequence", ""),
                subtype=payload.get("subtype", ""),
            )
            self.send_json(result)
        except (ValueError, SequenceValidationError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception:
            self.send_json(
                {"error": "예측 중 서버 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."},
                500,
            )
