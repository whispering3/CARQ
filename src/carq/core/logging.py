"""Logging estruturado com IDs de correlação e formatação JSON para observabilidade."""

import json
import logging
import logging.config
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from carq.core.config import settings

# ID de correlação por tarefa assíncrona — seguro para requisições concorrentes
_correlation_id_var: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

# Atributos padrão do LogRecord que NÃO devem ser promovidos a campos JSON.
# Calculado uma vez na importação para eficiência.
_STANDARD_LOG_ATTRS: frozenset[str] = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime"}


class CorrelationIDFilter(logging.Filter):
    """Adiciona ID de correlação aos registros de log para rastreamento de requisições.

    Usa contextvars.ContextVar para que cada tarefa assíncrona (requisição) tenha seu próprio
    ID de correlação isolado — sem vazamento entre requisições em ambientes assíncronos.
    """

    @staticmethod
    def set_correlation_id(correlation_id: Optional[str]) -> None:
        """Define o ID de correlação para o contexto assíncrono atual."""
        _correlation_id_var.set(correlation_id)

    @staticmethod
    def get_correlation_id() -> str:
        """Obtém ou gera o ID de correlação para o contexto assíncrono atual."""
        cid = _correlation_id_var.get()
        if cid is None:
            cid = str(uuid4())
            _correlation_id_var.set(cid)
        return cid

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = self.get_correlation_id()
        return True


class JSONFormatter(logging.Formatter):
    """Formata logs como JSON estruturado para produção."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", ""),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Coleta campos extras: qualquer atributo não-padrão adicionado via logger.xxx(extra={...})
        # O framework de logging define cada chave de `extra` diretamente no LogRecord,
        # não como um sub-dicionário — portanto hasattr(record, "extra") sempre será False.
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_ATTRS and not key.startswith("_"):
                log_data[key] = value

        return json.dumps(log_data, default=str)


class TextFormatter(logging.Formatter):
    """Formata logs como texto legível para desenvolvimento."""

    def format(self, record: logging.LogRecord) -> str:
        correlation_id = getattr(record, "correlation_id", "")
        correlation_str = f"[{correlation_id}] " if correlation_id else ""

        message = f"{correlation_str}{record.levelname:8} {record.name:30} {record.getMessage()}"

        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"

        return message


def _get_log_format() -> dict[str, Any]:
    formatter = settings.logging.format
    log_level = settings.logging.level

    base_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {},
        "filters": {
            "correlation_id": {
                "()": "carq.core.logging.CorrelationIDFilter",
            }
        },
        "handlers": {
            "default": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "default",
                "filters": ["correlation_id"],
                "stream": "ext://sys.stdout",
            }
        },
        "loggers": {
            "carq": {
                "level": log_level,
                "handlers": ["default"],
                "propagate": False,
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["default"],
        },
    }

    if formatter == "json":
        base_config["formatters"]["default"] = {
            "()": "carq.core.logging.JSONFormatter",
        }
    else:
        base_config["formatters"]["default"] = {
            "()": "carq.core.logging.TextFormatter",
        }

    if settings.logging.file:
        log_file = Path(settings.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        base_config["handlers"]["file"] = {
            "level": log_level,
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "default",
            "filters": ["correlation_id"],
        }
        base_config["loggers"]["carq"]["handlers"].append("file")
        base_config["root"]["handlers"].append("file")

    return base_config


def setup_logging(level: Optional[str] = None, json_format: Optional[bool] = None) -> "LoggerAdapter":
    """Inicializa o sistema de logging."""
    config = _get_log_format()
    if level:
        config["root"]["level"] = level
        config["loggers"]["carq"]["level"] = level
        config["handlers"]["default"]["level"] = level
    if json_format is not None and not json_format:
        config["formatters"]["default"] = {
            "()": "carq.core.logging.TextFormatter",
        }
    logging.config.dictConfig(config)
    return get_logger("carq")


def get_logger(name: str) -> logging.LoggerAdapter:
    """Retorna um logger com suporte a ID de correlação."""
    logger = logging.getLogger(name)
    return LoggerAdapter(logger)


class LoggerAdapter(logging.LoggerAdapter):
    """Adaptador para injetar contexto extra nos logs."""

    def __init__(self, logger: logging.Logger, extra: Optional[dict] = None):
        super().__init__(logger, extra or {})
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        """Injeta o contexto self.extra em cada registro de log."""
        existing_extra = kwargs.get("extra", {}) or {}
        kwargs["extra"] = {**self.extra, **existing_extra}
        return msg, kwargs

    def with_extra(self, **kwargs) -> "LoggerAdapter":
        """Retorna um novo adaptador com contexto extra adicional."""
        new_extra = {**self.extra, **kwargs}
        return LoggerAdapter(self.logger, new_extra)


# Inicializa o logging na importação
try:
    setup_logging()
except Exception as e:
    import sys
    print(f"WARNING: logging setup failed: {e}", file=sys.stderr)
