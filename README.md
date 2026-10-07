# Medical Image Analysis Assistant

An end-to-end **medical computer vision and AI engineering project** for chest X-ray analysis using Deep Learning, Explainable AI, FastAPI, and Docker.


## Project Overview

The system takes a chest X-ray image and processes it through a complete AI inference pipeline:

```text
Chest X-Ray
     ↓
Image Validation
     ↓
Image Preprocessing
     ↓
DenseNet121
     ↓
Prediction + Confidence
     ↓
Class Probabilities
     ↓
Grad-CAM Explainability
     ↓
FastAPI
     ↓
Docker
     ↓
Cloud Deployment

The model performs three-class classification:

* COVID-19
* Non-COVID Infection
* Normal

>Dataset
The project uses the **COVID-QU-Ex Dataset**, which contains chest X-ray images and lung segmentation masks.

The dataset was used for research, preprocessing, validation, model training, and evaluation.


The classification model is based on **pretrained DenseNet121** using transfer learning.


The inference pipeline performs:

* Image validation
* Grayscale conversion
* Resize to `224 × 224`
* Pixel normalization using `1/255`
* RGB channel conversion for DenseNet121


The project includes:

* Training accuracy
* Validation accuracy
* Training and validation loss
* Classification report
* Confusion matrix
* Prediction probabilities


The application uses **Grad-CAM** to generate visual explanations for model predictions.

```text
Original X-Ray
      +
Grad-CAM Heatmap
      +
Grad-CAM Overlay
```

Grad-CAM provides an interpretable visualization of image regions that contribute to the model prediction.

It should be considered an explainability aid and **not a clinical proof or diagnostic localization method**.


The trained model is exposed through a production-oriented FastAPI application.

Main endpoints:

```text
GET  /api/v1/health
POST /api/v1/predict
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

The prediction endpoint provides:

* Predicted class
* Confidence
* Class probabilities
* Model information
* Grad-CAM information
* Generated vi
```


The complete inference application is available as a Docker image.

> Pull the Docker Image

```bash
docker pull haseeb8184/medical-image-analysis-api:1.0.0
```

> Run the Container

```bash
docker run --name medical-api -p 8000:8000 haseeb8184/medical-image-analysis-api:1.0.0
```

Then open:

```text
http://localhost:8000/docs
```

For complete Docker instructions, see:

```text
DOCKER_IMAGE.md
```

## Technology Stack

**Machine Learning**

* Python
* TensorFlow / Keras
* DenseNet121
* NumPy
* OpenCV
* Pillow

**Explainability**

* Grad-CAM

**Backend**

* FastAPI
* Uvicorn
* Pydantic

**Deployment**

* Docker
* Docker Hub
* AWS-ready architecture

**Development**

* Jupyter Notebook
* VS Code
* Git
* GitHub

## Architecture

```text
                   Chest X-Ray
                       │
                       ▼
                ┌──────────────┐
                │ Image        │
                │ Validation   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ Preprocessing│
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │ DenseNet121  │
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
        Prediction          Grad-CAM
              │                 │
              └────────┬────────┘
                       ▼
                   FastAPI
                       │
                       ▼
                    Docker
                       │
                       ▼
                  AWS Cloud

```

## Future Roadmap

* AWS cloud deployment
* Public API endpoint
* HTTPS
* Authentication and API security
* Monitoring and logging
* Model versioning
* Calibration and uncertainty estimation
* Frontend application

## Portfolio

This project demonstrates an end-to-end AI engineering workflow:

```text
Data
 ↓
Computer Vision
 ↓
Deep Learning
 ↓
Model Evaluation
 ↓
Explainable AI
 ↓
FastAPI
 ↓
Docker
 ↓
Cloud Deployment


The objective is to demonstrate how a trained computer vision model can be transformed into a **usable, explainable, and deployable AI application**.

## Connect

If you find this project useful or interesting, feel free to explore the repository, try the Docker image, or connect with me to discuss **AI engineering, computer vision, medical imaging, and production ML systems**.

**If you like the project, consider giving the repository a star.**
