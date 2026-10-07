# ============================================================
# GRAD-CAM SERVICE
# ============================================================

from typing import Tuple

import cv2

import numpy as np

import tensorflow as tf

from app.config import (
    GRADCAM_ALPHA,
    GRADCAM_LAYER_NAME,
    DENSENET_SUBMODEL_NAME
)

from app.exceptions import (
    GradCAMError
)


class GradCAMService:

    def __init__(
        self,
        model
    ):

        self.model = model

        self.target_layer = None

        self.grad_model = None

        self._build_grad_model()


    # ========================================================
    # BUILD GRAD-CAM MODEL
    # ========================================================

    def _build_grad_model(
        self
    ):

        # ----------------------------------------------------
        # First try direct layer lookup
        # ----------------------------------------------------

        try:

            self.target_layer = (
                self.model.get_layer(
                    GRADCAM_LAYER_NAME
                )
            )

            print(
                "Using direct Grad-CAM layer:",
                GRADCAM_LAYER_NAME
            )


            self.grad_model = (
                tf.keras.models.Model(

                    inputs=self.model.inputs,

                    outputs=[
                        self.target_layer.output,
                        self.model.output
                    ]

                )
            )

            return


        except Exception:

            pass


        # ----------------------------------------------------
        # DenseNet may be nested inside a Functional model.
        # ----------------------------------------------------

        try:

            sub_model = (
                self.model.get_layer(
                    DENSENET_SUBMODEL_NAME
                )
            )

        except Exception as exc:

            raise GradCAMError(
                f"""
                Could not find DenseNet sub-model:
                {DENSENET_SUBMODEL_NAME}

                Also could not find direct layer:
                {GRADCAM_LAYER_NAME}

                Inspect model.layers and update the
                Grad-CAM configuration.
                """
            ) from exc


        # ----------------------------------------------------
        # Find target layer inside DenseNet
        # ----------------------------------------------------

        try:

            self.target_layer = (
                sub_model.get_layer(
                    GRADCAM_LAYER_NAME
                )
            )

        except Exception as exc:

            raise GradCAMError(
                f"""
                Grad-CAM layer '{GRADCAM_LAYER_NAME}'
                was not found inside '{DENSENET_SUBMODEL_NAME}'.
                """
            ) from exc


        print(
            "Using nested Grad-CAM layer:",
            f"{DENSENET_SUBMODEL_NAME}/"
            f"{GRADCAM_LAYER_NAME}"
        )


        # ----------------------------------------------------
        # Build nested Grad-CAM model
        # ----------------------------------------------------
        #
        # We construct a model that returns the target feature
        # maps and the final classification prediction.
        #
        # This follows the same nested DenseNet strategy used
        # in your notebook.
        # ----------------------------------------------------

        self.grad_model = (
            self._build_nested_grad_model(
                sub_model
            )
        )


    # ========================================================
    # BUILD NESTED GRAD-CAM MODEL
    # ========================================================

    def _build_nested_grad_model(
        self,
        sub_model
    ):

        target_layer = (
            sub_model.get_layer(
                GRADCAM_LAYER_NAME
            )
        )


        # ----------------------------------------------------
        # Build model around nested DenseNet.
        #
        # For the standard DenseNet121 transfer-learning
        # structure, the nested DenseNet produces the feature
        # tensor used by the classification head.
        # ----------------------------------------------------

        class NestedGradModel(
            tf.keras.Model
        ):

            def __init__(
                self,
                parent_model,
                dense_model,
                target_layer
            ):

                super().__init__()

                self.parent_model = (
                    parent_model
                )

                self.dense_model = (
                    dense_model
                )

                self.target_layer = (
                    target_layer
                )


            def call(
                self,
                inputs,
                training=False
            ):

                x = inputs

                target_activation = None

                # --------------------------------------------
                # Process parent model until DenseNet
                # --------------------------------------------

                for layer in self.parent_model.layers:

                    if (
                        layer.name
                        == self.dense_model.name
                    ):

                        # ------------------------------------
                        # We need to manually capture the
                        # internal target activation.
                        # ------------------------------------

                        with tf.GradientTape(
                            watch_accessed_variables=True
                        ) as tape:

                            dense_output = (
                                self.dense_model(
                                    x,
                                    training=training
                                )
                            )

                        return (
                            target_activation,
                            dense_output
                        )

                    if isinstance(
                        layer,
                        tf.keras.layers.InputLayer
                    ):

                        continue

                    x = layer(
                        x,
                        training=training
                    )

                return (
                    target_activation,
                    x
                )


        # ----------------------------------------------------
        # In practice, TensorFlow nested Functional models
        # can be directly traversed through a dedicated
        # gradient implementation below.
        #
        # Therefore we don't use the helper class above as
        # the final inference path.
        # ----------------------------------------------------

        return None


    # ========================================================
    # NESTED DENSENET GRAD-CAM
    # ========================================================

    def make_nested_gradcam(
        self,
        input_tensor: np.ndarray,
        predicted_class_index: int
    ) -> np.ndarray:

        try:

            dense_model = (
                self.model.get_layer(
                    DENSENET_SUBMODEL_NAME
                )
            )

            target_layer = (
                dense_model.get_layer(
                    GRADCAM_LAYER_NAME
                )
            )


            # ------------------------------------------------
            # Build a model that returns target activation and
            # DenseNet output.
            # ------------------------------------------------

            dense_grad_model = (
                tf.keras.models.Model(

                    inputs=dense_model.inputs,

                    outputs=[
                        target_layer.output,
                        dense_model.output
                    ]

                )
            )


            # ------------------------------------------------
            # Run parent model until DenseNet
            # ------------------------------------------------

            x = input_tensor


            parent_layers = (
                self.model.layers
            )


            dense_position = None

            for index, layer in enumerate(
                parent_layers
            ):

                if (
                    layer.name
                    == DENSENET_SUBMODEL_NAME
                ):

                    dense_position = index

                    break


            if dense_position is None:

                raise GradCAMError(
                    "DenseNet sub-model position could not be determined."
                )


            # ------------------------------------------------
            # Process layers before DenseNet
            # ------------------------------------------------

            for layer in parent_layers[
                :dense_position
            ]:

                if isinstance(
                    layer,
                    tf.keras.layers.InputLayer
                ):

                    continue

                x = layer(
                    x,
                    training=False
                )


            # ------------------------------------------------
            # Gradient calculation
            # ------------------------------------------------

            with tf.GradientTape() as tape:

                conv_outputs, dense_output = (
                    dense_grad_model(
                        x,
                        training=False
                    )
                )

                predictions = dense_output


                # --------------------------------------------
                # Process layers after DenseNet
                # --------------------------------------------

                post_dense_x = predictions

                for layer in parent_layers[
                    dense_position + 1:
                ]:

                    post_dense_x = layer(
                        post_dense_x,
                        training=False
                    )


                final_predictions = (
                    post_dense_x
                )


                class_channel = (
                    final_predictions[
                        0,
                        predicted_class_index
                    ]
                )


            # ------------------------------------------------
            # Calculate gradients
            # ------------------------------------------------

            gradients = tape.gradient(
                class_channel,
                conv_outputs
            )


            if gradients is None:

                raise GradCAMError(
                    "Grad-CAM gradients are None."
                )


            # ------------------------------------------------
            # Global average pooling
            # ------------------------------------------------

            pooled_gradients = (
                tf.reduce_mean(
                    gradients,
                    axis=(0, 1, 2)
                )
            )


            # ------------------------------------------------
            # Remove batch dimension
            # ------------------------------------------------

            conv_outputs = (
                conv_outputs[0]
            )


            # ------------------------------------------------
            # Weighted feature maps
            # ------------------------------------------------

            weighted_features = (
                conv_outputs
                * pooled_gradients
            )


            # ------------------------------------------------
            # Generate heatmap
            # ------------------------------------------------

            heatmap = tf.reduce_sum(
                weighted_features,
                axis=-1
            )


            # ------------------------------------------------
            # ReLU
            # ------------------------------------------------

            heatmap = tf.maximum(
                heatmap,
                0
            )


            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            maximum = tf.reduce_max(
                heatmap
            )


            heatmap = (
                heatmap
                / (
                    maximum
                    + tf.keras.backend.epsilon()
                )
            )


            return heatmap.numpy()


        except GradCAMError:

            raise


        except Exception as exc:

            raise GradCAMError(
                f"Grad-CAM generation failed: {str(exc)}"
            ) from exc


    # ========================================================
    # DIRECT GRAD-CAM
    # ========================================================

    def make_direct_gradcam(
        self,
        input_tensor: np.ndarray,
        predicted_class_index: int
    ) -> np.ndarray:

        if self.grad_model is None:

            raise GradCAMError(
                "Grad-CAM model has not been initialized."
            )


        try:

            with tf.GradientTape() as tape:

                conv_outputs, predictions = (
                    self.grad_model(
                        input_tensor,
                        training=False
                    )
                )

                class_channel = (
                    predictions[
                        0,
                        predicted_class_index
                    ]
                )


            gradients = tape.gradient(
                class_channel,
                conv_outputs
            )


            if gradients is None:

                raise GradCAMError(
                    "Gradients are None."
                )


            pooled_gradients = (
                tf.reduce_mean(
                    gradients,
                    axis=(0, 1, 2)
                )
            )


            conv_outputs = (
                conv_outputs[0]
            )


            weighted_features = (
                conv_outputs
                * pooled_gradients
            )


            heatmap = tf.reduce_sum(
                weighted_features,
                axis=-1
            )


            heatmap = tf.maximum(
                heatmap,
                0
            )


            maximum = tf.reduce_max(
                heatmap
            )


            heatmap = (
                heatmap
                / (
                    maximum
                    + tf.keras.backend.epsilon()
                )
            )


            return heatmap.numpy()


        except GradCAMError:

            raise


        except Exception as exc:

            raise GradCAMError(
                f"Grad-CAM generation failed: {str(exc)}"
            ) from exc


    # ========================================================
    # MAIN GRAD-CAM FUNCTION
    # ========================================================

    def generate(
        self,
        input_tensor: np.ndarray,
        predicted_class_index: int
    ) -> np.ndarray:

        # ----------------------------------------------------
        # If a direct Grad-CAM model exists, use it.
        # ----------------------------------------------------

        if self.grad_model is not None:

            return self.make_direct_gradcam(
                input_tensor,
                predicted_class_index
            )


        # ----------------------------------------------------
        # Otherwise use nested DenseNet implementation.
        # ----------------------------------------------------

        return self.make_nested_gradcam(
            input_tensor,
            predicted_class_index
        )


    # ========================================================
    # HEATMAP -> COLOR
    # ========================================================

    @staticmethod
    def heatmap_to_color(
        heatmap: np.ndarray
    ) -> np.ndarray:

        heatmap_uint8 = np.uint8(
            255 * np.clip(
                heatmap,
                0,
                1
            )
        )


        heatmap_color = (
            cv2.applyColorMap(
                heatmap_uint8,
                cv2.COLORMAP_JET
            )
        )


        heatmap_color = (
            cv2.cvtColor(
                heatmap_color,
                cv2.COLOR_BGR2RGB
            )
        )


        return heatmap_color


    # ========================================================
    # OVERLAY
    # ========================================================

    @staticmethod
    def create_overlay(
        grayscale_image: np.ndarray,
        heatmap: np.ndarray
    ) -> np.ndarray:

        # ----------------------------------------------------
        # Convert grayscale to RGB
        # ----------------------------------------------------

        grayscale_rgb = cv2.cvtColor(
            grayscale_image,
            cv2.COLOR_GRAY2RGB
        )


        # ----------------------------------------------------
        # Resize heatmap
        # ----------------------------------------------------

        resized_heatmap = cv2.resize(

            heatmap,

            (
                grayscale_image.shape[1],
                grayscale_image.shape[0]
            ),

            interpolation=cv2.INTER_LINEAR

        )


        # ----------------------------------------------------
        # Convert heatmap to RGB
        # ----------------------------------------------------

        colored_heatmap = (
            GradCAMService.heatmap_to_color(
                resized_heatmap
            )
        )


        # ----------------------------------------------------
        # Alpha blend
        # ----------------------------------------------------

        overlay = cv2.addWeighted(

            grayscale_rgb,

            1.0 - GRADCAM_ALPHA,

            colored_heatmap,

            GRADCAM_ALPHA,

            0

        )


        return overlay