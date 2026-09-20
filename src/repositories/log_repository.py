from repositories.Database import Database
from repositories.schemas.schema import ErrorLogger
from models.model import Error
from utils.Exceptions.errorcodes import database_error


class LogRepository:
    """Persists ErrorLogger rows to the database."""

    def __init__(self) -> None:
        self.db = Database()

    async def save_error_log(self, error: Error) -> None:
        """Insert an error log record using a proper async context manager session."""
        try:
            async with self.db.get_session() as session:
                log = ErrorLogger(
                    error_message=error.error_message,
                    function_name=error.function_name,
                    file_name=error.file_name,
                )
                session.add(log)
                # commit is handled by Database.get_session on clean exit
        except Exception as e:
            raise database_error(str(e))
