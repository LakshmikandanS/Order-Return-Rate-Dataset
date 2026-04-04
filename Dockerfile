FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir shap optuna fastapi uvicorn pydantic

# Copy project files
COPY data/ /app/data/
COPY models/ /app/models/
COPY scripts/ /app/scripts/
COPY preprocessing/ /app/preprocessing/
COPY api/ /app/api/
COPY web/ /app/web/

EXPOSE 8001
EXPOSE 8000

# Script to start both frontend dashboard and FastAPI backend
CMD ["sh", "-c", "cd web && python -m http.server 8000 & cd /app && uvicorn api.main:app --host 0.0.0.0 --port 8001"]
