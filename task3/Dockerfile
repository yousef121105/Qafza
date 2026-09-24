FROM python:3.10-slim

WORKDIR /app

COPY requirements/requirements-docker.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY data ./data
COPY config ./config
COPY mlflow-docker.db ./mlflow.db
COPY mlruns ./mlruns

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
