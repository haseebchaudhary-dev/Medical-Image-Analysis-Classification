# ============================================================
# MODEL SERVICE
# ============================================================

from pathlib import Path
from typing import Tuple

import numpy as np

import tensorflow as tf

from tensorflow.keras.models import load_model

from app.config import (
    CLASS_NAMES,
    INDEX_TO_CLASS,
    MODEL_NAME,
    MODEL_PATH
)

from app.exceptions import (
    ModelNotLoadedError
)


class ModelService:

    def __init__(
        self
    ):

        self.model = None

        self.grad_model = None

        self.last_conv_layer = None

        self.load_model()


    # ========================================================
    # LOAD MODEL
    # ========================================================

    def load_model(
        self
    ):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"""
                Model file was not found.

                Expected location:
                {MODEL_PATH}

                Place your trained model.keras file inside:

                fastapi_app/models/

                and rename it to:

                model.keras
                """
            )


        print(
            f"Loading model from: {MODEL_PATH}"
        )


        self.model = load_model(
            MODEL_PATH,
            compile=False
        )


        print(
            "Model loaded successfully."
        )

        print(
            "Input shape:",
            self.model.input_shape
        )

        print(
            "Output shape:",
            self.model.output_shape
        )


        # ----------------------------------------------------
        # Validate output dimensions
        # ----------------------------------------------------

        if (
            self.model.output_shape[-1]
            != len(CLASS_NAMES)
        ):

            raise RuntimeError(
                "Model output dimension does not match "
                "the configured class count."
            )


    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(
        self,
        input_tensor: np.ndarray
    ) -> Tuple[
        np.ndarray,
        int,
        str,
        float
    ]:

        if self.model is None:

            raise ModelNotLoadedError(
                "Classification model is not loaded."
            )


        predictions = self.model.predict(
            input_tensor,
            verbose=0
        )


        # ----------------------------------------------------
        # Handle model output
        # ----------------------------------------------------

        probabilities = np.asarray(
            predictions[0]
        ).astype(
            np.float32
        )


        # ----------------------------------------------------
        # Some models output logits instead of probabilities.
        #
        # Your trained classifier should output probabilities.
        #
        # This safety check converts values to a probability
        # distribution if necessary.
        # ----------------------------------------------------

        if (

            np.any(
                probabilities < 0
            )

            or

            not np.isclose(
                probabilities.sum(),
                1.0,
                atol=1e-3
            )

        ):

            probabilities = tf.nn.softmax(
                probabilities
            ).numpy()


        predicted_index = int(
            np.argmax(
                probabilities
            )
        )


        predicted_class = (
            INDEX_TO_CLASS[
                predicted_index
            ]
        )


        confidence = float(
            probabilities[
                predicted_index
            ]
        )


        return (
            probabilities,
            predicted_index,
            predicted_class,
            confidence
        )