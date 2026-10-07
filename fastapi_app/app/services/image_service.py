# ============================================================
# IMAGE VALIDATION + PREPROCESSING SERVICE
# ============================================================

from io import BytesIO
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from PIL import Image

from fastapi import UploadFile

from app.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_UPLOAD_SIZE_BYTES
)

from app.exceptions import (
    ImageTooLargeError,
    InvalidImageError
)


# ============================================================
# READ UPLOAD
# ============================================================

async def read_uploaded_image(
    file: UploadFile
) -> bytes:

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise InvalidImageError(
            "Uploaded file does not have a valid filename."
        )

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_EXTENSIONS:

        raise InvalidImageError(
            "Invalid file extension. "
            "Only image files are accepted."
        )


    # --------------------------------------------------------
    # Validate declared content type
    # --------------------------------------------------------

    if file.content_type not in ALLOWED_CONTENT_TYPES:

        raise InvalidImageError(
            "Invalid content type. "
            "Only JPEG, PNG, BMP and TIFF images are accepted."
        )


    # --------------------------------------------------------
    # Read bytes
    # --------------------------------------------------------

    image_bytes = await file.read()


    # --------------------------------------------------------
    # File size validation
    # --------------------------------------------------------

    if not image_bytes:

        raise InvalidImageError(
            "Uploaded file is empty."
        )

    if len(image_bytes) > MAX_UPLOAD_SIZE_BYTES:

        raise ImageTooLargeError(
            "Uploaded image exceeds the maximum "
            "allowed file size."
        )


    return image_bytes


# ============================================================
# VERIFY IMAGE CONTENT
# ============================================================

def decode_image(
    image_bytes: bytes
) -> Image.Image:

    try:

        image = Image.open(
            BytesIO(image_bytes)
        )

        # Force PIL to actually decode the image.
        image.load()

    except Exception as exc:

        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from exc


    # --------------------------------------------------------
    # Verify image format
    # --------------------------------------------------------

    if image.format is None:

        raise InvalidImageError(
            "Unable to determine image format."
        )


    return image


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

def convert_to_grayscale(
    image: Image.Image
) -> np.ndarray:

    grayscale = image.convert(
        "L"
    )

    grayscale_array = np.array(
        grayscale
    )

    if grayscale_array.size == 0:

        raise InvalidImageError(
            "Image contains no pixel data."
        )


    return grayscale_array


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    grayscale_image: np.ndarray
) -> Tuple[
    np.ndarray,
    np.ndarray
]:

    # --------------------------------------------------------
    # Resize to DenseNet input dimensions
    # --------------------------------------------------------

    resized = cv2.resize(

        grayscale_image,

        (
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        ),

        interpolation=cv2.INTER_AREA

    )


    # --------------------------------------------------------
    # Convert grayscale -> RGB
    #
    # DenseNet121 was trained with 3-channel input.
    #
    # Therefore we replicate the grayscale channel three times.
    # --------------------------------------------------------

    rgb_image = np.stack(
        [
            resized,
            resized,
            resized
        ],
        axis=-1
    )


    # --------------------------------------------------------
    # Convert to float32
    # --------------------------------------------------------

    normalized = (
        rgb_image.astype(
            np.float32
        )
        / 255.0
    )


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    input_tensor = np.expand_dims(
        normalized,
        axis=0
    )


    return (
        resized,
        input_tensor
    )


# ============================================================
# COMPLETE IMAGE PIPELINE
# ============================================================

def prepare_image(
    image_bytes: bytes
):

    image = decode_image(
        image_bytes
    )

    grayscale_image = (
        convert_to_grayscale(
            image
        )
    )

    resized_grayscale, input_tensor = (
        preprocess_image(
            grayscale_image
        )
    )

    return {
        "original_image": image,

        "grayscale_image":
            grayscale_image,

        "resized_grayscale":
            resized_grayscale,

        "input_tensor":
            input_tensor
    }