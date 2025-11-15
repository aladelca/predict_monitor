"""Aplicación FastAPI para exponer el modelo de predicción."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import PricePredictor


predictor = PricePredictor(auto_load=False)
app = FastAPI(
    title="Multimodal Price Predictor",
    description="API para inferir precios a partir de imagen + descripción",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    image_path: str = Field(..., description="Ruta local al archivo de imagen")
    description: str = Field(..., description="Descripción del producto")


class PredictResponse(BaseModel):
    predicted_price: float = Field(..., description="Precio estimado en la misma moneda del entrenamiento")


@app.on_event("startup")
def _load_predictor() -> None:
    predictor.load()


@app.get("/health", tags=["health"])
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["predictions"])
def predict_price(payload: PredictRequest) -> PredictResponse:
    try:
        prediction = predictor.predict(payload.image_path, payload.description)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - se reporta error genérico para el cliente
        raise HTTPException(status_code=500, detail="Error inesperado durante la predicción") from exc
    return PredictResponse(predicted_price=prediction)

