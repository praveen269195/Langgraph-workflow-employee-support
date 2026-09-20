from datetime import datetime, timezone
from typing import Optional, Dict

from pydantic import ValidationError

from models.model import APIResponse
from utils.Exceptions.errorcodes import ApplicationError, ErrorCode


class ResponseFormatter:
    """Formats all API responses into the standard envelope.

    Shape is identical to the previous project — do not change field names.
    """

    def format_success_response(
        self,
        code: int,
        message: str,
        data: Optional[Dict],
        request_id: str,
    ) -> APIResponse:
        """Standard 2xx response."""
        return APIResponse(
            code=code,
            status="success",
            message=message,
            data=data,
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def format_error_response(
        self,
        error: ApplicationError,
        request_id: str,
    ) -> APIResponse:
        """Standard error response from an ApplicationError."""
        return APIResponse(
            code=error.http_status,
            status="Error",
            message=error.message,
            error=error.to_dict()["error"],
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def format_validation_error_response(
        self,
        error: ValidationError,
        request_id: str,
    ) -> APIResponse:
        """Validation error response — surfaces per-field details."""
        details = {
            err["loc"][-1] if err["loc"] else "field": err["msg"]
            for err in error.errors()
        }
        return APIResponse(
            code=400,
            status="Error",
            message="Validation failed",
            error={
                "code": ErrorCode.VALIDATION_ERROR.value,
                "message": "Validation failed",
                "details": details,
            },
            request_id=request_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
