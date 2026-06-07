"""Minimal HTTP server for the ADC score predictor."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

from adc_model import ADCScoreModel, SequenceValidationError

STATIC_DIR = Path(__file__).resolve().parent / "public"
_model: ADCScoreModel | None = None
_model_lock = Lock()


def get_model() -> ADCScoreModel:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ADCScoreModel()
    return _model


class ADCRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/metadata":
            try:
                model = get_model()
                self._send_json(
                    {
                        "subtypes": model.subtypes,
                        "training_count": model.training_count,
                    },
                    HTTPStatus.OK,
                )
            except Exception as error:
                self._send_json(
                    {"error": f"모델을 준비하지 못했습니다: {error}"},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0 or content_length > 200_000:
                raise ValueError("요청 데이터의 크기가 올바르지 않습니다.")
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            result = get_model().predict(
                heavy_sequence=payload.get("heavy_sequence", ""),
                light_sequence=payload.get("light_sequence", ""),
                subtype=payload.get("subtype", ""),
            )
            self._send_json(result, HTTPStatus.OK)
        except (ValueError, SequenceValidationError, json.JSONDecodeError) as error:
            self._send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:
            self._send_json(
                {"error": f"예측 중 오류가 발생했습니다: {error}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ADC score web application.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("ADC 모델을 학습하고 있습니다...")
    model = get_model()
    print(f"학습 완료: {model.training_count}개 항체")

    server = ThreadingHTTPServer((args.host, args.port), ADCRequestHandler)
    print(f"웹 앱 실행: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
