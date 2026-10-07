# ============================================================
# APPLICATION EXCEPTIONS
# ============================================================


class InvalidImageError(Exception):

    def __init__(
        self,
        message: str
    ):

        self.message = message

        super().__init__(
            self.message
        )


class ImageTooLargeError(Exception):

    def __init__(
        self,
        message: str
    ):

        self.message = message

        super().__init__(
            self.message
        )


class ModelNotLoadedError(Exception):

    def __init__(
        self,
        message: str
    ):

        self.message = message

        super().__init__(
            self.message
        )


class GradCAMError(Exception):

    def __init__(
        self,
        message: str
    ):

        self.message = message

        super().__init__(
            self.message
        )