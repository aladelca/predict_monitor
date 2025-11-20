"""AWS Lambda handler para inferencia con el Multimodal Price Predictor."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict

from src.predict import PricePredictor


_predictor = PricePredictor(auto_load=False)


@dataclass
class LambdaResponse:
    status_code: int
    body: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statusCode": self.status_code,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(self.body),
        }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:  # noqa: D401 - firma dictada por Lambda
    """Punto de entrada para AWS Lambda."""

    try:
        payload = _extract_payload(event)
    except ValueError as exc:
        return LambdaResponse(HTTPStatus.BAD_REQUEST, {"message": str(exc)}).to_dict()

    image_url = payload.get("image_url") or payload.get("image_path")
    description = payload.get("description")

    if not isinstance(image_url, str) or not image_url.strip():
        return LambdaResponse(HTTPStatus.BAD_REQUEST, {"message": "`image_url` es obligatorio"}).to_dict()

    if not isinstance(description, str) or not description.strip():
        return LambdaResponse(HTTPStatus.UNPROCESSABLE_ENTITY, {"message": "`description` es obligatorio"}).to_dict()

    try:
        prediction = _predictor.predict(image_url.strip(), description.strip())
    except FileNotFoundError as exc:
        return LambdaResponse(HTTPStatus.BAD_REQUEST, {"message": str(exc)}).to_dict()
    except ValueError as exc:
        return LambdaResponse(HTTPStatus.UNPROCESSABLE_ENTITY, {"message": str(exc)}).to_dict()
    except Exception as exc:  # pragma: no cover - error inesperado
        return LambdaResponse(HTTPStatus.INTERNAL_SERVER_ERROR, {"message": "Error no controlado", "detail": str(exc)}).to_dict()

    return LambdaResponse(HTTPStatus.OK, {"predicted_price": prediction}).to_dict()


def _extract_payload(event: Dict[str, Any]) -> Dict[str, Any]:
    if not event:
        raise ValueError("Evento vacío")

    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if event.get("isBase64Encoded"):
            body = _decode_base64(body)
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = body or "{}"
            return json.loads(body)
        if isinstance(body, dict):
            return body
        raise ValueError("Formato de body no soportado")

    if isinstance(event, dict):
        return event

    raise ValueError("Evento no reconocido")


def _decode_base64(body: Any) -> bytes:
    if isinstance(body, str):
        return base64.b64decode(body)
    if isinstance(body, (bytes, bytearray)):
        return base64.b64decode(body)
    raise ValueError("Body codificado en base64 no es válido")
