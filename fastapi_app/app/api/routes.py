# ============================================================
# API ROUTES
# ============================================================

import base64
from io import BytesIO

from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile
)

from PIL import Image

from app.config import (
    API_VERSION,
    CLASS_NAMES,
    GRADCAM_LAYER_NAME,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MODEL_NAME
)

from app.exceptions import (
    GradCAMError,
    ImageTooLargeError,
    InvalidImageError,
    ModelNotLoadedError
)

from app.schemas import (
    HealthResponse
)

from app.services.image_service import (
    prepare_image,
    read_uploaded_image
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter()


# ============================================================
# DEPENDENCY PLACEHOLDERS
# ============================================================

model_service = None
gradcam_service = None


def initialize_services(
    model_service_instance,
    gradcam_service_instance
):

    global model_service
    global gradcam_service

    model_service = model_service_instance
    gradcam_service = gradcam_service_instance


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"]
)
def health_check():

    return HealthResponse(

        status="healthy",

        model_loaded=(
            model_service is not None
            and model_service.model is not None
        ),

        model_name=MODEL_NAME,

        version=API_VERSION

    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@router.get(
    "/",
    tags=["System"]
)
def root():

    return {

        "application":
            "Medical Image Analysis Assistant",

        "version":
            API_VERSION,

        "model":
            MODEL_NAME,

        "task":
            "Chest X-Ray Classification + Grad-CAM",

        "classes":
            CLASS_NAMES,

        "documentation":
            "/docs"

    }


# ============================================================
# IMAGE → BASE64
# ============================================================

def image_to_base64(
    image: Image.Image
) -> str:

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return base64.b64encode(
        buffer.read()
    ).decode("utf-8")


# ============================================================
# PREDICT ENDPOINT
# ============================================================

@router.post(
    "/predict",
    tags=["Prediction"]
)
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate service availability
    # --------------------------------------------------------

    if model_service is None:

        raise HTTPException(
            status_code=503,
            detail="Prediction service is unavailable."
        )

    if gradcam_service is None:

        raise HTTPException(
            status_code=503,
            detail="Grad-CAM service is unavailable."
        )


    # --------------------------------------------------------
    # Validate and read upload
    # --------------------------------------------------------

    try:

        image_bytes = await read_uploaded_image(
            file
        )

    except ImageTooLargeError as exc:

        raise HTTPException(
            status_code=413,
            detail=exc.message
        )

    except InvalidImageError as exc:

        raise HTTPException(
            status_code=400,
            detail=exc.message
        )


    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    try:

        image_data = prepare_image(
            image_bytes
        )

    except InvalidImageError as exc:

        raise HTTPException(
            status_code=400,
            detail=exc.message
        )


    input_tensor = image_data[
        "input_tensor"
    ]

    resized_grayscale = image_data[
        "resized_grayscale"
    ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        (
            probabilities,
            predicted_index,
            predicted_class,
            confidence

        ) = model_service.predict(
            input_tensor
        )

    except ModelNotLoadedError as exc:

        raise HTTPException(
            status_code=503,
            detail=exc.message
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail="Model inference failed."
        ) from exc


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    try:

        heatmap = gradcam_service.generate(
            input_tensor,
            predicted_index
        )

        colored_heatmap = (
            gradcam_service.heatmap_to_color(
                heatmap
            )
        )

        overlay = (
            gradcam_service.create_overlay(
                resized_grayscale,
                heatmap
            )
        )

    except GradCAMError as exc:

        raise HTTPException(
            status_code=500,
            detail=exc.message
        )


    # --------------------------------------------------------
    # Convert images
    # --------------------------------------------------------

    grayscale_image = Image.fromarray(
        resized_grayscale
    )

    heatmap_image = Image.fromarray(
        colored_heatmap
    )

    overlay_image = Image.fromarray(
        overlay
    )


    # --------------------------------------------------------
    # Encode images
    # --------------------------------------------------------

    grayscale_base64 = image_to_base64(
        grayscale_image
    )

    heatmap_base64 = image_to_base64(
        heatmap_image
    )

    overlay_base64 = image_to_base64(
        overlay_image
    )


    # --------------------------------------------------------
    # Probability dictionary
    # --------------------------------------------------------

    probability_dict = {

        "COVID-19":
            float(probabilities[0]),

        "Non-COVID Infection":
            float(probabilities[1]),

        "Normal":
            float(probabilities[2])

    }


    # --------------------------------------------------------
    # JSON RESPONSE
    # --------------------------------------------------------

    return {

        "success": True,

        "prediction": {

            "predicted_class":
                predicted_class,

            "confidence":
                float(confidence),

            "probabilities":
                probability_dict

        },

        "model": {

            "name":
                MODEL_NAME,

            "input_size":
                f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",

            "normalization":
                "1/255"

        },

        "explainability": {

            "method":
                "Grad-CAM",

            "target_class":
                predicted_class,

            "layer":
                GRADCAM_LAYER_NAME

        },

        "images": {

            "grayscale":
                grayscale_base64,

            "gradcam_heatmap":
                heatmap_base64,

            "gradcam_overlay":
                overlay_base64

        },

        "image_format":
            "PNG",

        "warning":
            (
                "This output is intended for research and "
                "decision-support purposes and must not be "
                "treated as an autonomous clinical diagnosis."
            )

    }