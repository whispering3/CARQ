"""Unit tests for REST API (Phase 6).

Tests cover:
- Request/response models validation
- Authentication
- Endpoint functionality
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from carq.api.models import (
    EmbedRequest,
    EmbedResponse,
    SearchRequest,
    SearchResponse,
    BatchEmbedRequest,
    BatchEmbedResponse,
    StatsResponse,
    EmbeddingModelEnum,
)
from carq.api.auth import APIKeyAuth, verify_api_key
from carq.api.router import create_router, EmbeddingAPI


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def embed_request():
    """Create embed request."""
    return EmbedRequest(
        text="This is a test document.",
        model=EmbeddingModelEnum.OPENAI_3_SMALL,
    )


@pytest.fixture
def search_request():
    """Create search request."""
    return SearchRequest(
        query="machine learning",
        model=EmbeddingModelEnum.OPENAI_3_SMALL,
        limit=10,
        similarity_threshold=0.7,
    )


@pytest.fixture
def batch_embed_request():
    """Create batch embed request."""
    return BatchEmbedRequest(
        texts=["Text 1", "Text 2", "Text 3"],
        model=EmbeddingModelEnum.OPENAI_3_SMALL,
    )


@pytest.fixture
def mock_dispatcher():
    """Create mocked dispatcher."""
    return AsyncMock()


@pytest.fixture
def mock_vector_store():
    """Create mocked vector store."""
    return AsyncMock()


@pytest.fixture
def mock_cache():
    """Create mocked cache."""
    return AsyncMock()


@pytest.fixture
def embedding_api(mock_dispatcher, mock_vector_store, mock_cache):
    """Create embedding API instance."""
    return EmbeddingAPI(
        dispatcher=mock_dispatcher,
        vector_store=mock_vector_store,
        cache=mock_cache,
    )


@pytest.fixture
def app(mock_dispatcher, mock_vector_store, mock_cache):
    """Create FastAPI test app."""
    from unittest.mock import patch
    from carq.api.auth import _load_valid_keys

    _load_valid_keys.cache_clear()
    with patch.dict("os.environ", {"CARQ_API_KEYS": "sk-test-key"}):
        _load_valid_keys.cache_clear()
        test_app = FastAPI()
        router = create_router(mock_dispatcher, mock_vector_store, mock_cache)
        test_app.include_router(router)
        yield test_app
    _load_valid_keys.cache_clear()


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# ============================================================================
# REQUEST/RESPONSE MODEL TESTS
# ============================================================================


def test_embed_request_valid(embed_request):
    """Test valid embed request."""
    assert embed_request.text == "This is a test document."
    assert embed_request.model == EmbeddingModelEnum.OPENAI_3_SMALL


def test_embed_request_validation():
    """Test embed request validation."""
    # Valid
    req = EmbedRequest(text="test")
    assert req.text == "test"

    # Invalid: empty text
    with pytest.raises(ValueError):
        EmbedRequest(text="")

    # Invalid: too long
    with pytest.raises(ValueError):
        EmbedRequest(text="x" * 20000)


def test_embed_response_creation():
    """Test embed response creation."""
    response = EmbedResponse(
        text="test",
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        tokens_used=10,
        cost_usd=0.00001,
    )

    assert response.text == "test"
    assert len(response.embedding) == 1536
    assert response.cached is False


def test_search_request_validation():
    """Test search request validation."""
    # Valid
    req = SearchRequest(query="test")
    assert req.limit == 10
    assert req.similarity_threshold == 0.7

    # Invalid: limit too high
    with pytest.raises(ValueError):
        SearchRequest(query="test", limit=200)

    # Invalid: threshold out of range
    with pytest.raises(ValueError):
        SearchRequest(query="test", similarity_threshold=1.5)


def test_batch_embed_request_validation():
    """Test batch embed request validation."""
    # Valid
    req = BatchEmbedRequest(texts=["text1", "text2"])
    assert len(req.texts) == 2

    # Invalid: no texts
    with pytest.raises(ValueError):
        BatchEmbedRequest(texts=[])

    # Invalid: too many texts
    with pytest.raises(ValueError):
        BatchEmbedRequest(texts=[f"text{i}" for i in range(200)])


def test_search_result_creation():
    """Test search result creation."""
    result = SearchResponse(
        query="test",
        results=[],
        count=0,
        search_latency_ms=45.2,
    )

    assert result.query == "test"
    assert result.count == 0


def test_batch_embed_response_creation():
    """Test batch embed response creation."""
    response = BatchEmbedResponse(
        embeddings=[],
        count=0,
        total_tokens=0,
        total_cost_usd=0.0,
        batch_latency_ms=100.0,
    )

    assert response.count == 0
    assert response.total_cost_usd == 0.0


def test_stats_response_creation():
    """Test stats response creation."""
    response = StatsResponse(
        embeddings_generated=100,
        total_tokens_used=5000,
        total_cost_usd=0.50,
        avg_cost_per_embedding=0.005,
        cache_hit_rate=0.75,
        cached_embeddings=500,
        stored_vectors=1000,
    )

    assert response.embeddings_generated == 100
    assert response.cache_hit_rate == 0.75


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_api_key_auth_valid():
    """Test valid API key authentication."""
    auth = APIKeyAuth(valid_keys={"sk-test-key"})

    result = await auth(authorization="Bearer sk-test-key")
    assert result == "sk-test-key"


@pytest.mark.asyncio
async def test_api_key_auth_invalid():
    """Test invalid API key authentication."""
    from fastapi import HTTPException

    auth = APIKeyAuth(valid_keys={"sk-test-key"})

    with pytest.raises(HTTPException):
        await auth(authorization="Bearer sk-invalid-key")


@pytest.mark.asyncio
async def test_api_key_auth_missing():
    """Test missing API key."""
    from fastapi import HTTPException

    auth = APIKeyAuth(valid_keys={"sk-test-key"})

    with pytest.raises(HTTPException):
        await auth(authorization=None)


@pytest.mark.asyncio
async def test_api_key_auth_format():
    """Test invalid authorization format."""
    from fastapi import HTTPException

    auth = APIKeyAuth(valid_keys={"sk-test-key"})

    with pytest.raises(HTTPException):
        await auth(authorization="InvalidFormat sk-test-key")


# ============================================================================
# ENDPOINT TESTS
# ============================================================================


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_embed_endpoint_missing_auth(client):
    """Test embed endpoint without authentication."""
    request = {"text": "test"}
    response = client.post("/api/v1/embed", json=request)
    assert response.status_code == 401


def test_embed_endpoint_invalid_request(client):
    """Test embed endpoint with invalid request."""
    headers = {"X-API-Key": "sk-test-key"}
    request = {"text": ""}  # Invalid: empty
    response = client.post("/api/v1/embed", json=request, headers=headers)
    assert response.status_code == 422  # Validation error


def test_search_endpoint_missing_auth(client):
    """Test search endpoint without authentication."""
    request = {"query": "test"}
    response = client.post("/api/v1/search", json=request)
    assert response.status_code == 401


def test_stats_endpoint_missing_auth(client):
    """Test stats endpoint without authentication."""
    response = client.get("/api/v1/stats")
    assert response.status_code == 401


# ============================================================================
# EMBEDDING API TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_embedding_api_cache_hit(embedding_api, embed_request):
    """Test cache hit in embedding API."""
    # Mock cache hit
    embedding_api.cache.get.return_value = [0.1] * 1536

    response = await embedding_api.embed_single(embed_request, "sk-test")

    assert response.cached is True
    assert response.cost_usd == 0.0
    embedding_api.dispatcher.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_api_cache_miss(embedding_api, embed_request):
    """Test cache miss in embedding API."""
    # Mock cache miss
    embedding_api.cache.get.return_value = None

    # Mock dispatcher
    from carq.embedding.embedding_dispatcher import EmbeddingResult

    mock_result = EmbeddingResult(
        text="test",
        embedding=[0.1] * 1536,
        model="text-embedding-3-small",
        tokens_used=10,
        cost_usd=0.00001,
    )
    embedding_api.dispatcher.embed.return_value = mock_result

    response = await embedding_api.embed_single(embed_request, "sk-test")

    assert response.cached is False
    assert response.cost_usd > 0.0
    embedding_api.cache.set.assert_called()


@pytest.mark.asyncio
async def test_embedding_api_error_handling(embedding_api, embed_request):
    """Test error handling in embedding API."""
    from fastapi import HTTPException
    from carq.embedding.embedding_dispatcher import EmbeddingError

    # Mock error
    embedding_api.cache.get.return_value = None
    embedding_api.dispatcher.embed.side_effect = EmbeddingError("API error")

    with pytest.raises(HTTPException):
        await embedding_api.embed_single(embed_request, "sk-test")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_model_enum_values():
    """Test embedding model enum values."""
    assert EmbeddingModelEnum.OPENAI_3_LARGE.value == "text-embedding-3-large"
    assert EmbeddingModelEnum.OPENAI_3_SMALL.value == "text-embedding-3-small"
    assert EmbeddingModelEnum.OPENAI_ADA.value == "text-embedding-ada-002"


def test_request_json_serialization(embed_request):
    """Test request serialization to JSON."""
    json_str = embed_request.model_dump_json()
    assert "text" in json_str
    assert "model" in json_str


def test_response_json_serialization():
    """Test response serialization to JSON."""
    response = EmbedResponse(
        text="test",
        embedding=[0.1, 0.2],
        model="text-embedding-3-small",
        tokens_used=10,
        cost_usd=0.00001,
    )

    json_str = response.model_dump_json()
    assert "text" in json_str
    assert "embedding" in json_str
    assert "cost_usd" in json_str


def test_stats_response_structure():
    """Test stats response structure."""
    stats = StatsResponse(
        embeddings_generated=0,
        total_tokens_used=0,
        total_cost_usd=0.0,
        avg_cost_per_embedding=0.0,
        cache_hit_rate=0.0,
        cached_embeddings=0,
        stored_vectors=0,
    )

    assert stats.embeddings_generated == 0
    assert stats.cache_hit_rate == 0.0
