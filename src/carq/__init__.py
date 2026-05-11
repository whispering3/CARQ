"""
CARQ - Context-Aware RAG Processing Queue
Orquestrador Empresarial de Ingestão RAG com Rate Limiting e Resiliência a Falhas
"""

__version__ = "0.1.0"
__author__ = "Danilo"
__description__ = "Context-Aware RAG Processing Queue"

from carq.core.config import settings
from carq.core.logging import get_logger

__all__ = ["settings", "get_logger"]
