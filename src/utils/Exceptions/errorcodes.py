# utils/Exceptions/errorcodes.py
from enum import Enum
from typing import Optional, Dict, Any

# -------------------------------------------------
# Error Codes
# -------------------------------------------------
class ErrorCode(Enum):
    DATABASE_ERROR          = "ZAP_SQL_005"
    VALIDATION_ERROR        = "ZAP_VAL_022"
    NOT_FOUND               = "ZAP_GEN_030"
    UNAUTHORIZED            = "ZAP_AUTH_018"
    FORBIDDEN               = "ZAP_AUTH_026"
    INTERNAL_SERVER_ERROR   = "ZAP_SRV_006"

# -------------------------------------------------
# HTTP Status Mapping
# -------------------------------------------------
HTTP_STATUS_MAP = {
    ErrorCode.DATABASE_ERROR:        500,
    ErrorCode.VALIDATION_ERROR:      400,
    ErrorCode.NOT_FOUND:             404,
    ErrorCode.UNAUTHORIZED:          401,
    ErrorCode.FORBIDDEN:             403,
    ErrorCode.INTERNAL_SERVER_ERROR: 500,
}

# -------------------------------------------------
# Custom Application Exception
# -------------------------------------------------
class ApplicationError(Exception):
    def __init__(
        self,
        error_code: ErrorCode,              
        message: str,                    
        details: Optional[Dict[str, Any]] = None
    ):
        self.code        = error_code.value
        self.http_status = HTTP_STATUS_MAP.get(error_code, 500)
        self.message     = message
        self.details     = details or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": {
                "code":     self.code,
                "message":  self.message,
                "details":  self.details
            }
        }

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def validation_error(field: str, reason: str) -> ApplicationError:
    return ApplicationError(
        error_code=ErrorCode.VALIDATION_ERROR,
        message=f"Invalid value for '{field}'",
        details={"reason": reason}
    )

def not_found_error(resource: str) -> ApplicationError:
    return ApplicationError(
        error_code=ErrorCode.NOT_FOUND,
        message=f"{resource} not found",
    )

def database_error(operation: str) -> ApplicationError:
    return ApplicationError(
        error_code=ErrorCode.DATABASE_ERROR,
        message=f"Database {operation} failed",
    )

def unauthorized_error() -> ApplicationError:
    return ApplicationError(
        error_code=ErrorCode.UNAUTHORIZED,
        message="User is not authenticated",
    )

def forbidden_error() -> ApplicationError:
    return ApplicationError(
        error_code=ErrorCode.FORBIDDEN,
        message="Access denied",

    )

def client_error(message: str) -> ApplicationError:
    return ApplicationError(
        error_code=ErrorCode.INTERNAL_SERVER_ERROR,
        message=message,
    )
    
