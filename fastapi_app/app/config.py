# ============================================================
# CONFIGURATION
# ============================================================

from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

FASTAPI_ROOT = APP_DIR.parent

MODEL_DIR = FASTAPI_ROOT / "models"

# ------------------------------------------------------------
# MODEL PATH
# ------------------------------------------------------------
#
# PLACE YOUR MODEL HERE:
#
# fastapi_app/
#     models/
#         model.keras
#
# If you use another filename, update MODEL_FILENAME.
# ------------------------------------------------------------

MODEL_FILENAME = "model.keras"

MODEL_PATH = MODEL_DIR / MODEL_FILENAME


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

MODEL_NAME = "DenseNet121"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_SIZE = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)

NORMALIZATION_FACTOR = 1.0 / 255.0


# ------------------------------------------------------------
# CLASS CONFIGURATION
# ------------------------------------------------------------

CLASS_NAMES = [
    "COVID-19",
    "Non-COVID Infection",
    "Normal"
]

CLASS_TO_INDEX = {
    "COVID-19": 0,
    "Non-COVID Infection": 1,
    "Normal": 2
}

INDEX_TO_CLASS = {
    0: "COVID-19",
    1: "Non-COVID Infection",
    2: "Normal"
}


# ------------------------------------------------------------
# GRAD-CAM CONFIGURATION
# ------------------------------------------------------------

# This is the DenseNet121 layer used in your notebook.
#
# If your saved model has a different internal layer name,
# update this value.
# ------------------------------------------------------------

DENSENET_SUBMODEL_NAME = "densenet121"

GRADCAM_LAYER_NAME = "conv5_block16_concat"

GRADCAM_ALPHA = 0.40


# ------------------------------------------------------------
# UPLOAD VALIDATION
# ------------------------------------------------------------

# Maximum accepted upload size.
#
# 10 MB is reasonable for an API image endpoint.
# ------------------------------------------------------------

MAX_UPLOAD_SIZE_MB = 10

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)


# ------------------------------------------------------------
# SECURITY
# ------------------------------------------------------------

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff"
}


# ------------------------------------------------------------
# API INFORMATION
# ------------------------------------------------------------

API_TITLE = (
    "Medical Image Analysis Assistant API"
)

API_DESCRIPTION = """
Production-oriented medical chest X-ray inference API.

The API performs:
- image validation
- grayscale conversion
- resizing
- normalization
- DenseNet121 classification
- class probability estimation
- Grad-CAM explainability

This system is intended for research and decision-support
purposes and is NOT an autonomous clinical diagnostic system.
"""

API_VERSION = "1.0.0"