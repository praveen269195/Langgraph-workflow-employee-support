from models.model import Error
from repositories.log_repository import LogRepository
from utils.Exceptions.errorcodes import client_error


class ErrorLogHandler:
    """Persists error events to the database via LogRepository."""

    def __init__(self) -> None:
        self.repo = LogRepository()

    async def log_error_details(
        self,
        file_name: str,
        func_name: str,
        message: str,
    ) -> None:
        try:
            error = Error(
                function_name=func_name,
                file_name=file_name,
                error_message=message,
            )
            await self.repo.save_error_log(error)
        except Exception as e:
            # Always raise so the caller (Logger) knows DB persistence failed.
            raise client_error(str(e))
