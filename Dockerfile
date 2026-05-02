FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY models ./models
COPY configs ./configs
COPY reports ./reports
COPY data/samples ./data/samples

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -e .

EXPOSE 8000

CMD ["uvicorn", "scisum_qwen.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
