"""
Exceções personalizadas para o sistema CARQ.
Organizadas por categoria para tratamento limpo de erros.
"""

from typing import Any, Optional


class CARQException(Exception):
    """Exceção base para todos os erros do CARQ."""

    def __init__(
        self,
        message: str,
        code: str = "CARQ_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Converte a exceção em dicionário para resposta da API."""
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "details": self.details,
            }
        }




class ConfigurationError(CARQException):
    """Configuração inválida ou incompleta."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="CONFIG_ERROR", status_code=500, details=details)


class DatabaseConfigError(ConfigurationError):
    """Configuração do banco de dados é inválida."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Database config error: {message}", details=details)


class EmbeddingConfigError(ConfigurationError):
    """Configuração do serviço de embedding é inválida."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(f"Embedding config error: {message}", details=details)




class DatabaseError(CARQException):
    """Operação de banco de dados falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="DATABASE_ERROR", status_code=500, details=details)


class ConnectionPoolError(DatabaseError):
    """Pool de conexões esgotado ou com falha."""

    def __init__(self, message: str = "Connection pool unavailable", details: Optional[dict] = None):
        super().__init__(message, details=details)


class TransactionError(DatabaseError):
    """Transação de banco de dados falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)


class IdempotencyError(DatabaseError):
    """Processamento duplicado detectado (esperado, não fatal)."""

    def __init__(self, message: str = "Task already processed", details: Optional[dict] = None):
        super().__init__(message, code="IDEMPOTENCY_CONFLICT", status_code=409, details=details)




class QueueError(CARQException):
    """Operação de fila falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="QUEUE_ERROR", status_code=500, details=details)


class TaskNotFoundError(QueueError):
    """Tarefa não encontrada na fila."""

    def __init__(self, task_id: str):
        super().__init__(
            f"Task not found: {task_id}",
            code="TASK_NOT_FOUND",
            status_code=404,
            details={"task_id": task_id},
        )


class TaskAlreadyProcessingError(QueueError):
    """Tarefa já está sendo processada."""

    def __init__(self, task_id: str):
        super().__init__(
            f"Task already processing: {task_id}",
            code="TASK_ALREADY_PROCESSING",
            status_code=409,
            details={"task_id": task_id},
        )




class RateLimitError(CARQException):
    """Limite de taxa excedido."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int = 60,
        details: Optional[dict] = None,
    ):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            message,
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details,
        )
        self.retry_after = retry_after


class CircuitBreakerOpenError(CARQException):
    """Circuit breaker aberto, serviço indisponível."""

    def __init__(
        self,
        message: str,
        retry_after: int = 60,
        details: Optional[dict] = None,
    ):
        details = details or {}
        details["retry_after"] = retry_after
        super().__init__(
            message,
            code="CIRCUIT_BREAKER_OPEN",
            status_code=503,
            details=details,
        )
        self.retry_after = retry_after




class ProcessingError(CARQException):
    """Processamento de documento falhou."""

    def __init__(self, message: str, code: str = "PROCESSING_ERROR", details: Optional[dict] = None):
        super().__init__(message, code=code, status_code=400, details=details)


class PDFParsingError(ProcessingError):
    """Falha na análise do arquivo PDF."""

    def __init__(self, filename: str, reason: str, details: Optional[dict] = None):
        details = details or {}
        details["filename"] = filename
        details["reason"] = reason
        super().__init__(
            f"Failed to parse PDF {filename}: {reason}",
            code="PDF_PARSING_ERROR",
            details=details,
        )


class ChunkingError(ProcessingError):
    """Segmentação de documento falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            details=details,
        )


class EmbeddingError(ProcessingError):
    """Geração de embedding falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            code="EMBEDDING_ERROR",
            details=details,
        )


class VectorInsertionError(ProcessingError):
    """Inserção de vetor no banco de dados falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            code="VECTOR_INSERTION_ERROR",
            details=details,
        )




class ValidationError(CARQException):
    """Validação de entrada falhou."""

    def __init__(self, message: str, field: Optional[str] = None, details: Optional[dict] = None):
        details = details or {}
        if field:
            details["field"] = field
        super().__init__(
            message,
            code="VALIDATION_ERROR",
            status_code=422,
            details=details,
        )


class DocumentValidationError(ValidationError):
    """Validação de documento falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)


class ChunkValidationError(ValidationError):
    """Validação de chunk falhou."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, details=details)




class ExternalServiceError(CARQException):
    """Serviço externo (OpenAI, etc.) falhou."""

    def __init__(
        self,
        service: str,
        message: str,
        status_code: int = 502,
        details: Optional[dict] = None,
    ):
        details = details or {}
        details["service"] = service
        super().__init__(
            f"{service} error: {message}",
            code="EXTERNAL_SERVICE_ERROR",
            status_code=status_code,
            details=details,
        )


class OpenAIError(ExternalServiceError):
    """Erro na API da OpenAI."""

    def __init__(self, message: str, status_code: int = 502, details: Optional[dict] = None):
        super().__init__("OpenAI", message, status_code=status_code, details=details)




class RetryableError(CARQException):
    """Operação falhou mas pode ser repetida."""

    def __init__(
        self,
        message: str,
        retry_after: int = 5,
        max_retries: int = 3,
        code: str = "RETRYABLE_ERROR",
        details: Optional[dict] = None,
    ):
        details = details or {}
        details["retry_after"] = retry_after
        details["max_retries"] = max_retries
        super().__init__(
            message,
            code=code,
            status_code=503,
            details=details,
        )
        self.retry_after = retry_after
        self.max_retries = max_retries


class TaskTimeoutError(RetryableError):
    """Execução da tarefa excedeu o tempo limite."""

    def __init__(self, task_id: str, timeout_seconds: int, details: Optional[dict] = None):
        details = details or {}
        details["task_id"] = task_id
        details["timeout_seconds"] = timeout_seconds
        super().__init__(
            f"Task {task_id} exceeded timeout of {timeout_seconds}s",
            code="TASK_TIMEOUT",
            details=details,
        )




class IntegrityError(CARQException):
    """Violação de integridade de dados."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(
            message,
            code="INTEGRITY_ERROR",
            status_code=409,
            details=details,
        )


class DuplicateResourceError(IntegrityError):
    """Recurso já existe."""

    def __init__(self, resource_type: str, resource_id: str, details: Optional[dict] = None):
        details = details or {}
        details["resource_type"] = resource_type
        details["resource_id"] = resource_id
        super().__init__(
            f"{resource_type} already exists: {resource_id}",
            code="DUPLICATE_RESOURCE",
            details=details,
        )


# Aliases para compatibilidade retroativa
ValidationException = ValidationError


class PDFParseError(CARQException):
    """Erro de análise de PDF com mensagem única."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message, code="PDF_PARSE_ERROR", status_code=400, details=details)
