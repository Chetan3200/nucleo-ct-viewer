class AppError(Exception):
    """A safe, user-facing error. Never put secrets or raw upstream text here."""

    def __init__(self, message: str, status_code: int = 400, code: str = "invalid_input"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
