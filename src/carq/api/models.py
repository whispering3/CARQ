"""Modelos Pydantic de requisição/resposta para a API REST."""

from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingModelEnum(str, Enum):
    """Modelos de embedding disponíveis via API."""

    OPENAI_3_LARGE = "text-embedding-3-large"
    OPENAI_3_SMALL = "text-embedding-3-small"
    OPENAI_ADA = "text-embedding-ada-002"


class EmbedRequest(BaseModel):
    """Requisição para gerar um único embedding."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "text": "This is a sample document for embedding.",
            "model": "text-embedding-3-small",
        }
    })

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Texto para gerar embedding",
    )
    model: EmbeddingModelEnum = Field(
        default=EmbeddingModelEnum.OPENAI_3_SMALL,
        description="Modelo de embedding a usar",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Metadados opcionais",
    )


class EmbedResponse(BaseModel):
    """Resposta com embedding."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "text": "This is a sample document for embedding.",
            "embedding": [0.1, 0.2, 0.3],
            "model": "text-embedding-3-small",
            "tokens_used": 10,
            "cost_usd": 0.00001,
            "cached": False,
        }
    })

    text: str = Field(..., description="Texto original")
    embedding: List[float] = Field(..., description="Vetor de embedding")
    model: str = Field(..., description="Modelo utilizado")
    tokens_used: int = Field(..., description="Tokens consumidos")
    cost_usd: float = Field(..., description="Custo em USD")
    cached: bool = Field(
        default=False,
        description="Indica se o embedding foi obtido do cache",
    )


class SearchResult(BaseModel):
    """Resultado único de busca."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "chunk_id": "chunk_123",
            "text": "Document text here...",
            "similarity_score": 0.87,
            "metadata": {"source": "pdf_1"},
        }
    })

    chunk_id: str = Field(..., description="Identificador do chunk")
    text: str = Field(..., description="Texto do chunk")
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similaridade cosseno (0,0-1,0)",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Metadados do chunk",
    )


class SearchRequest(BaseModel):
    """Requisição para busca semântica."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "machine learning models",
            "model": "text-embedding-3-small",
            "limit": 10,
            "similarity_threshold": 0.7,
        }
    })

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Texto de consulta para busca",
    )
    model: EmbeddingModelEnum = Field(
        default=EmbeddingModelEnum.OPENAI_3_SMALL,
        description="Modelo de embedding a usar",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Número máximo de resultados",
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similaridade mínima (0,0-1,0)",
    )


class SearchResponse(BaseModel):
    """Resposta com resultados de busca."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "machine learning models",
            "results": [
                {
                    "chunk_id": "chunk_123",
                    "text": "ML models...",
                    "similarity_score": 0.87,
                }
            ],
            "count": 1,
            "search_latency_ms": 45.2,
        }
    })

    query: str = Field(..., description="Consulta original")
    results: List[SearchResult] = Field(..., description="Resultados da busca")
    count: int = Field(..., description="Número de resultados")
    search_latency_ms: float = Field(..., description="Tempo de busca em milissegundos")


class BatchEmbedRequest(BaseModel):
    """Requisição para gerar múltiplos embeddings."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "texts": [
                "First document",
                "Second document",
            ],
            "model": "text-embedding-3-small",
        }
    })

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Texts to embed",
    )
    model: EmbeddingModelEnum = Field(
        default=EmbeddingModelEnum.OPENAI_3_SMALL,
        description="Embedding model to use",
    )


class BatchEmbedResponse(BaseModel):
    """Resposta com embeddings em lote."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "embeddings": [
                {
                    "text": "First document",
                    "embedding": [0.1, 0.2],
                    "model": "text-embedding-3-small",
                    "tokens_used": 5,
                    "cost_usd": 0.000001,
                }
            ],
            "count": 1,
            "total_tokens": 5,
            "total_cost_usd": 0.000001,
            "batch_latency_ms": 150.5,
        }
    })

    embeddings: List[EmbedResponse] = Field(
        ...,
        description="Generated embeddings",
    )
    count: int = Field(..., description="Number of embeddings")
    total_tokens: int = Field(..., description="Total tokens used")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    batch_latency_ms: float = Field(
        ...,
        description="Batch processing time",
    )


class StatsResponse(BaseModel):
    """Resposta com estatísticas do sistema."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "embeddings_generated": 1000,
            "total_tokens_used": 50000,
            "total_cost_usd": 1.23,
            "avg_cost_per_embedding": 0.00123,
            "cache_hit_rate": 0.75,
            "cached_embeddings": 500,
            "stored_vectors": 1000,
        }
    })

    embeddings_generated: int = Field(..., description="Total embeddings generated")
    total_tokens_used: int = Field(..., description="Total tokens consumed")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    avg_cost_per_embedding: float = Field(
        ...,
        description="Average cost per embedding",
    )
    cache_hit_rate: float = Field(..., description="Cache hit rate (0.0-1.0)")
    cached_embeddings: int = Field(..., description="Number of cached embeddings")
    stored_vectors: int = Field(..., description="Vectors in database")


class ErrorResponse(BaseModel):
    """Resposta de erro."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "error": "Invalid API key",
            "detail": "Authentication failed",
            "status_code": 401,
        }
    })

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(
        default=None,
        description="Error details",
    )
    status_code: int = Field(..., description="HTTP status code")


class DocumentType(str, Enum):
    """Tipos de documento suportados para ingestão."""

    PDF = "pdf"
    TEXT = "text"
    URL = "url"


class IngestRequest(BaseModel):
    """Requisição para ingerir um documento no pipeline RAG."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "source_uri": "https://example.com/document.pdf",
            "document_type": "pdf",
            "priority": 0,
            "attributes": {"author": "Jane Smith", "category": "research"},
        }
    })

    source_uri: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="URI of the document (URL, file path, or unique identifier)",
    )
    document_type: DocumentType = Field(
        default=DocumentType.PDF,
        description="Type of the document",
    )
    content: Optional[str] = Field(
        default=None,
        max_length=1_000_000,
        description="Raw text content (required when document_type=text)",
    )
    priority: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Processing priority (0=normal, 10=highest)",
    )
    attributes: Optional[dict] = Field(
        default=None,
        description="Arbitrary metadata to attach to the document",
    )


class IngestResponse(BaseModel):
    """Resposta após aceitação do documento para ingestão."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "document_id": "550e8400-e29b-41d4-a716-446655440000",
            "status": "pending",
            "message": "Document accepted for processing",
            "task_id": "660e8400-e29b-41d4-a716-446655440001",
        }
    })

    document_id: str = Field(..., description="UUID of the created document")
    status: str = Field(..., description="Initial document status")
    message: str = Field(..., description="Human-readable status message")
    task_id: Optional[str] = Field(
        default=None,
        description="UUID of the initial processing task",
    )


class DocumentStatusResponse(BaseModel):
    """Resposta com status atual de processamento do documento."""

    document_id: str = Field(..., description="Document UUID")
    status: str = Field(..., description="Current processing status")
    document_type: str = Field(..., description="Document type")
    source_uri: str = Field(..., description="Document source URI")
    chunk_count: int = Field(..., description="Number of chunks created")
    embedding_count: int = Field(..., description="Number of embeddings generated")
    progress_percentage: float = Field(..., description="Processing progress (0-100)")
    error_message: Optional[str] = Field(default=None, description="Error if failed")
