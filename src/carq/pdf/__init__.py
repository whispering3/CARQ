"""Módulo de Processamento de PDF.

Gerencia extração assíncrona de PDFs, segmentação semântica e validação de qualidade.
"""

from .chunk_validator import ChunkValidator, ValidationResult
from .pdf_parser import PDFParser, PDFParsingError
from .semantic_chunker import ChunkingConfig, SemanticChunker

__all__ = [
    "PDFParser",
    "PDFParsingError",
    "SemanticChunker",
    "ChunkingConfig",
    "ChunkValidator",
    "ValidationResult",
]
