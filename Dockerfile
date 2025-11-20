# Imagen base para AWS Lambda (runtime Python 3.11)
FROM public.ecr.aws/lambda/python:3.11

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SENTENCE_TRANSFORMERS_HOME=/var/task/cache/sentence-transformers \
    TRANSFORMERS_CACHE=/var/task/cache/transformers \
    HF_HOME=/var/task/cache/huggingface \
    TORCH_HOME=/var/task/cache/torch \
    PYTHONPATH="/var/task:${PYTHONPATH}"

WORKDIR /var/task

COPY requirements-lambda.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p "$SENTENCE_TRANSFORMERS_HOME" "$TRANSFORMERS_CACHE" "$HF_HOME" "$TORCH_HOME"

# Descarga anticipada de pesos SentenceTransformer y ResNet para evitar dependencias de red en runtime
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet18, ResNet18_Weights
SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
PY

COPY model.pth ./model.pth
COPY src ./src
COPY lambda_function.py ./lambda_function.py

CMD ["lambda_function.lambda_handler"]
