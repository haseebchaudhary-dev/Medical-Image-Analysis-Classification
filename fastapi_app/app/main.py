# ============================================================
# FASTAPI APPLICATION ENTRY POINT
# ============================================================

from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    Request
)

from fastapi.responses import JSONResponse

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    MODEL_NAME
)

from app.api.routes import (
    initialize_services,
    router
)

from app.services.model_service import (
    ModelService
)

from app.services.gradcam_service import (
    GradCAMService
)


# ============================================================
# GLOBAL SERVICES
# ============================================================

model_service = None

gradcam_service = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================
#
# The model is loaded ONCE when FastAPI starts.
#
# We do NOT load model.keras for every request.
#
# This is extremely important for production inference.
# ============================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI
):

    global model_service

    global gradcam_service


    print(
        "\n"
        + "=" * 70
    )

    print(
        "Starting Medical Image Analysis Assistant API"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Load classification model
    # --------------------------------------------------------

    model_service = ModelService()


    # --------------------------------------------------------
    # Build Grad-CAM service
    # --------------------------------------------------------

    gradcam_service = (
        GradCAMService(
            model_service.model
        )
    )


    # --------------------------------------------------------
    # Initialize API services
    # --------------------------------------------------------

    initialize_services(

        model_service,

        gradcam_service

    )


    print(
        "\nModel:",
        MODEL_NAME
    )

    print(
        "Model loaded successfully."
    )

    print(
        "API is ready."
    )

    print(
        "=" * 70,
        "\n"
    )


    yield


    # ========================================================
    # SHUTDOWN
    # ========================================================

    print(
        "\nShutting down API..."
    )


    model_service = None

    gradcam_service = None


# ============================================================
# CREATE FASTAPI APP
# ============================================================

app = FastAPI(

    title=API_TITLE,

    description=API_DESCRIPTION,

    version=API_VERSION,

    lifespan=lifespan

)


# ============================================================
# INCLUDE ROUTES
# ============================================================

app.include_router(
    router,
    prefix="/api/v1"
)


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(
    Exception
)
async def global_exception_handler(
    request: Request,
    exc: Exception
):

    print(
        "Unhandled exception:",
        str(exc)
    )


    return JSONResponse(

        status_code=500,

        content={

            "success": False,

            "error": {

                "type":
                    "InternalServerError",

                "message":
                    "An unexpected server error occurred."

            }

        }

    )


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "Medical Image Analysis Assistant API",
        "status": "running",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/health",
    tags=["System"]
)
def root_health():

    return {

        "status":
            "healthy",

        "service":
            "Medical Image Analysis Assistant",

        "model":
            MODEL_NAME,

        "version":
            API_VERSION

    }