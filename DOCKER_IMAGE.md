# Docker Image

The Medical Image Analysis Assistant inference API is available as a Docker image through Docker Hub.

## Docker Image

```text
haseeb8184/medical-image-analysis-api:1.0.0
```

## Pull the Image

Make sure Docker Desktop is running, then execute:

```bash
docker pull haseeb8184/medical-image-analysis-api:1.0.0
```

## Run the Container

```bash
docker run --name medical-api -p 8000:8000 haseeb8184/medical-image-analysis-api:1.0.0
```

The API will then be available at:

```text
http://localhost:8000
```

## API Documentation

Open:

```text
http://localhost:8000/docs
```

This opens the FastAPI Swagger UI where the API endpoints can be tested.

## Health Check

Endpoint:

```text
GET /api/v1/health
```

Example:

```bash
curl http://localhost:8000/api/v1/health
```

## Prediction

Endpoint:

```text
POST /api/v1/predict
```

The endpoint accepts a chest X-ray image and returns:

* Predicted class
* Prediction confidence
* Class probabilities
* Model metadata
* Grad-CAM information
* Grad-CAM visualization images

Supported classes:

```text
COVID-19
Non-COVID Infection
Normal
```

## Stop the Container

```bash
docker stop medical-api
```

Remove the container:

```bash
docker rm medical-api
```

Or stop and remove it together:

```bash
docker rm -f medical-api
```

## Docker Workflow

```text
Docker Hub
    │
    │ docker pull
    ▼
Docker Image
    │
    │ docker run
    ▼
FastAPI Container
    │
    ├── TensorFlow
    ├── DenseNet121
    ├── Grad-CAM
    └── FastAPI
    │
    ▼
localhost:8000
    │
    ▼
Swagger / Prediction API
```

## Important Note

This Docker image is intended for **research and decision-support purposes**.

It is not an autonomous clinical diagnostic system and should not be used as a substitute for professional medical evaluation.

## Docker Hub

Docker image:

```text
haseeb8184/medical-image-analysis-api:1.0.0
```

Pull it with:

```bash
docker pull haseeb8184/medical-image-analysis-api:1.0.0
```

Explore the main project repository for the complete machine learning pipeline, evaluation, explainability, FastAPI implementation, and deployment architecture.
