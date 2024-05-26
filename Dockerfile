FROM python:3.12-bullseye

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install build dependencies.
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential cmake libeigen3-dev gcc curl libsndfile1 zlib1g-dev ffmpeg

# Setup venv for building all requirements and add it to path
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Run the web service on container startup. Here we use the gunicorn
# Each worker creates a full replication of the service, i.e. VRAM footprint of all loaded models
# Threads run on a per-worker basis
# TODO: Not sure if we will be using gunicorn for this
CMD exec gunicorn --bind :$PORT --workers 1 --threads 12 --timeout 0 main:app