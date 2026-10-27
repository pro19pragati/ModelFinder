"""Domain errors mapped to HTTP responses by the API layer."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentError(Exception):
    """Base class for expected, user-facing errors."""

    status_code = 400
    code = "error"

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code:
            self.code = code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"error": self.code, "message": self.message, "details": self.details}


class DatasetValidationError(AgentError):
    """Unsupported format, empty file, corrupt file, missing target, ..."""

    status_code = 400
    code = "dataset_validation_error"


class ResourceLimitError(AgentError):
    """Dataset exceeds a configured hard resource limit."""

    status_code = 413
    code = "resource_limit_exceeded"

    def __init__(self, message: str, violations: Optional[List[Dict[str, str]]] = None) -> None:
        super().__init__(
            message,
            code="resource_limit_exceeded",
            details={"violations": violations or []},
        )
        self.violations = violations or []


class NotFoundError(AgentError):
    status_code = 404
    code = "not_found"


class ConflictError(AgentError):
    """Requested action is not valid in the current state (e.g. already running)."""

    status_code = 409
    code = "conflict"


class TrainingError(AgentError):
    status_code = 422
    code = "training_error"
