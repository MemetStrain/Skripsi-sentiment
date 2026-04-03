FROM python:3.11-slim

# System deps for torch/numpy/statsmodels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY scheduler/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy scheduler source
COPY scheduler/ .

# Copy existing analysis scripts (used by scheduler)
COPY cpo/ /cpo/
COPY news/ /news/
COPY markov/ /markov/
COPY prediction/ /prediction/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "main.py"]
