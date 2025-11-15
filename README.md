# Multimodal Price Predictor

Servicio ligero para realizar inferencias con el modelo multimodal descrito en `machine_learning_multimodal.ipynb`. El proyecto expone un helper reutilizable (`PricePredictor`) y una API REST con FastAPI que recibe la ruta (local o remota) de la imagen más la descripción del producto y devuelve el precio estimado.

## Estructura principal

- `src/config.py`: rutas por defecto, selección de dispositivo y parámetros comunes de inferencia.
- `src/models.py`: definición del `MultimodalRegressor` (ResNet18 + MLP para texto) usado en entrenamiento e inferencia.
- `src/predict.py`: clase `PricePredictor`, encargada de inicializar el codificador de texto, normalizaciones de imagen y cargar `model.pth`. Soporta imágenes locales o URLs HTTP/HTTPS.
- `src/app.py`: aplicación FastAPI con endpoints `/health` y `/predict` que delegan en `PricePredictor`.

## Requisitos

- Python 3.11+ (se recomienda usar el `venv` local `.venv`).
- Pesos del modelo en la raíz del proyecto (`model.pth`).
- Dependencias principales: `torch`, `torchvision`, `sentence-transformers`, `pillow`, `fastapi`, `uvicorn`, `requests`.

Instalación sugerida:

```bash
python -m venv .venv
source .venv/bin/activate  # en macOS/Linux
pip install --upgrade pip
pip install torch torchvision sentence-transformers pillow fastapi uvicorn requests
```

## Uso del predictor en scripts

```python
from src.predict import PricePredictor

predictor = PricePredictor()
pred_local = predictor.predict("imagenes/0.png", "Monitor LG 24'' Full HD")
pred_remote = predictor.predict(
    "https://images.philips.com/is/image/philipsconsumer/...png",
    "Monitor Philips 27'' 4K"
)
print(pred_local, pred_remote)
```

La clase detecta automáticamente si `image_path` es una ruta local o una URL (`http/https`). Para URLs, descarga el binario en memoria usando `requests` (sin persistir archivos temporales) y aplica las mismas transformaciones usadas durante el entrenamiento.

Parámetros útiles de `PricePredictor`:

- `model_path`: ruta alternativa al archivo `.pth`.
- `text_model_name`: nombre del modelo SentenceTransformer a utilizar.
- `device`: dispositivo de ejecución (`cuda`, `mps`, `cpu`).
- `http_timeout` / `http_headers`: control sobre las peticiones a imágenes remotas.

## API REST con FastAPI

1. Activa el entorno virtual e instala dependencias (ver sección anterior).
2. Ejecuta el servidor:

   ```bash
   uvicorn src.app:app --reload
   ```

3. Endpoints:

   - `GET /health`: verificación simple (`{"status":"ok"}`).
   - `POST /predict`: cuerpo JSON con `image_path` y `description`.

Ejemplo de llamada:

```bash
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{
           "image_path": "https://images.philips.com/is/image/philipsconsumer/...png",
           "description": "Monitor Philips 27\" 4K UHD"
         }'
```

Respuesta:

```json
{
  "predicted_price": 1299.42
}
```

Puedes abrir `http://127.0.0.1:8000/docs` para explorar la documentación interactiva generada por FastAPI/Swagger.

## Buenas prácticas y notas

- El predictor fija las normalizaciones de imagen a los valores de ImageNet y utiliza `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, idéntico al cuaderno de entrenamiento.
- La carga del checkpoint contempla formatos guardados como diccionarios o modelos completos y maneja los cambios recientes de seguridad en `torch.load`.
- Si necesitas afinar el rendimiento, puedes inicializar `PricePredictor(auto_load=False)` y decidir manualmente cuándo invocar `load()` (por ejemplo, en workers asincrónicos).
- No se incluyen pruebas unitarias; se recomienda añadirlas si planeas desplegar en producción o automatizar regresiones.

