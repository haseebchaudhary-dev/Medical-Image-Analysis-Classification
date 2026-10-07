# ============================================================
# API RESPONSE SCHEMAS
# ============================================================

from typing import Dict

from pydantic import BaseModel, Field


# ------------------------------------------------------------
# CLASS PROBABILITIES
# ------------------------------------------------------------

class PredictionProbabilities(BaseModel):

    covid_19: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )

    non_covid_infection: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )

    normal: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )


# ------------------------------------------------------------
# PREDICTION RESPONSE
# ------------------------------------------------------------

class PredictionResponse(BaseModel):

    success: bool

    predicted_class: str

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0
    )

    probabilities: PredictionProbabilities

    model_name: str

    image_size: str

    normalization: str

    gradcam_layer: str

    message: str


# ------------------------------------------------------------
# HEALTH RESPONSE
# ------------------------------------------------------------

class HealthResponse(BaseModel):

    status: str

    model_loaded: bool

    model_name: str

    version: str