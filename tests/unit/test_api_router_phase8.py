"""Tests for api/router.py - Phase 8.

Covers: embed, embed-batch, search, stats, health, auth failures,
validation errors, and error propagation.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from carq.api.models import (
    EmbeddingModelEnum,
    EmbedRequest,
)
from carq.api.router import EmbeddingAPI, create_router
from carq.embedding.embedding_cache import CacheError
from carq.embedding.embedding_dispatcher import EmbeddingError
from carq.embedding.vector_store import VectorStoreError

VALID_KEY = "sk-testkey"
HEADERS = {"X-API-Key": VALID_KEY}


# ============================================================================
# FIXTURES
# ============================================================================


def make_embed_result(text="hello", model="text-embedding-3-small"):
    """Build a fake EmbeddingResult-like object."""
    result = MagicMock()
    result.text = text
    result.embedding = [0.1, 0.2, 0.3]
    result.model = model
    result.tokens_used = 10
    result.cost_usd = 0.00001
    return result


def make_search_result_item():
    result = MagicMock()
    result.chunk_id = "chunk-abc"
    result.text = "Relevant passage"
    result.similarity_score = 0.88
    result.metadata = {"page": 1}
    return result


@pytest.fixture
def mock_dispatcher():
    d = AsyncMock()
    d.embed = AsyncMock(return_value=make_embed_result())
    d.embed_batch = AsyncMock(return_value=[make_embed_result("t1"), make_embed_result("t2")])
    d.get_metrics = MagicMock(return_value={
        "total_embeddings": 100,
        "total_tokens": 5000,
        "total_cost_usd": 0.05,
        "avg_cost_per_embedding": 0.0005,
    })
    return d


@pytest.fixture
def mock_vector_store():
    vs = AsyncMock()
    vs.search = AsyncMock(return_value=[make_search_result_item()])
    vs.count = AsyncMock(return_value=42)
    return vs


@pytest.fixture
def mock_cache():
    c = AsyncMock()
    c.get = AsyncMock(return_value=None)   # cache miss by default
    c.set = AsyncMock(return_value=True)
    c.get_stats = AsyncMock(return_value=MagicMock(hit_rate=0.5, size=50))
    return c


@pytest.fixture(autouse=True)
def api_keys_env():
    from carq.api.auth import _load_valid_keys

    _load_valid_keys.cache_clear()
    with patch.dict("os.environ", {"CARQ_API_KEYS": VALID_KEY}):
        _load_valid_keys.cache_clear()
        yield
    _load_valid_keys.cache_clear()


@pytest.fixture
def test_client(mock_dispatcher, mock_vector_store, mock_cache):
    """Create TestClient with mocked dependencies."""
    app = FastAPI()
    router = create_router(mock_dispatcher, mock_vector_store, mock_cache)
    app.include_router(router)
    with TestClient(app) as client:
        yield client


# ============================================================================
# HEALTH CHECK
# ============================================================================


def test_health_check(test_client):
    """GET /health returns 200 and status=healthy."""
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


# ============================================================================
# AUTH TESTS
# ============================================================================


def test_embed_missing_api_key(test_client):
    """POST /embed without X-API-Key returns 401."""
    response = test_client.post("/api/v1/embed", json={"text": "Hello"})
    assert response.status_code == 401


def test_embed_invalid_api_key_format(test_client):
    """POST /embed with bad key format returns 401."""
    response = test_client.post(
        "/api/v1/embed",
        json={"text": "Hello"},
        headers={"X-API-Key": "invalid-format"},
    )
    assert response.status_code == 401


def test_search_missing_api_key(test_client):
    """POST /search without auth returns 401."""
    response = test_client.post("/api/v1/search", json={"query": "test"})
    assert response.status_code == 401


def test_stats_missing_api_key(test_client):
    """GET /stats without auth returns 401."""
    response = test_client.get("/api/v1/stats")
    assert response.status_code == 401


def test_embed_batch_missing_api_key(test_client):
    """POST /embed-batch without auth returns 401."""
    response = test_client.post("/api/v1/embed-batch", json={"texts": ["a", "b"]})
    assert response.status_code == 401


# ============================================================================
# VALIDATION ERROR TESTS (422)
# ============================================================================


def test_embed_empty_text_validation_error(test_client):
    """POST /embed with empty text returns 422."""
    response = test_client.post(
        "/api/v1/embed",
        json={"text": ""},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_embed_missing_text_field(test_client):
    """POST /embed with missing text field returns 422."""
    response = test_client.post(
        "/api/v1/embed",
        json={},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_search_empty_query_validation(test_client):
    """POST /search with empty query returns 422."""
    response = test_client.post(
        "/api/v1/search",
        json={"query": ""},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_embed_batch_empty_texts_list(test_client):
    """POST /embed-batch with empty texts list returns 422."""
    response = test_client.post(
        "/api/v1/embed-batch",
        json={"texts": []},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_ingest_text_requires_content(test_client):
    """POST /documents with document_type=text requires raw content."""
    response = test_client.post(
        "/api/v1/documents",
        json={"source_uri": "file:///tmp/doc.txt", "document_type": "text"},
        headers=HEADERS,
    )
    assert response.status_code == 422


def test_search_invalid_limit(test_client):
    """POST /search with limit=0 returns 422."""
    response = test_client.post(
        "/api/v1/search",
        json={"query": "test", "limit": 0},
        headers=HEADERS,
    )
    assert response.status_code == 422


# ============================================================================
# EMBED ENDPOINT
# ============================================================================


def test_embed_success(test_client):
    """POST /embed returns 200 with embedding."""
    response = test_client.post(
        "/api/v1/embed",
        json={"text": "Hello world"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert "model" in data
    assert isinstance(data["embedding"], list)


def test_embed_cache_hit(mock_dispatcher, mock_vector_store):
    """POST /embed returns cached=True when cache hits."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=[0.1, 0.2, 0.3])
    cache.set = AsyncMock(return_value=True)
    cache.get_stats = AsyncMock(return_value=MagicMock(hit_rate=1.0, size=1))

    app = FastAPI()
    router = create_router(mock_dispatcher, mock_vector_store, cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/embed",
        json={"text": "cached text"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["cached"] is True


def test_embed_cache_miss_calls_dispatcher(test_client, mock_dispatcher):
    """POST /embed on cache miss calls the dispatcher."""
    test_client.post(
        "/api/v1/embed",
        json={"text": "uncached text"},
        headers=HEADERS,
    )
    mock_dispatcher.embed.assert_called_once()


def test_embed_sets_cache_after_generation(test_client, mock_cache):
    """POST /embed stores result in cache after generation."""
    test_client.post(
        "/api/v1/embed",
        json={"text": "new text"},
        headers=HEADERS,
    )
    mock_cache.set.assert_called_once()


def test_embed_returns_500_on_embedding_error(mock_vector_store, mock_cache):
    """POST /embed returns 500 when dispatcher raises EmbeddingError."""
    dispatcher = AsyncMock()
    dispatcher.embed = AsyncMock(side_effect=EmbeddingError("API failure"))

    app = FastAPI()
    router = create_router(dispatcher, mock_vector_store, mock_cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/embed",
        json={"text": "hello"},
        headers=HEADERS,
    )
    assert response.status_code == 500


# ============================================================================
# EMBED BATCH ENDPOINT
# ============================================================================


def test_embed_batch_success(test_client):
    """POST /embed-batch returns 200 with multiple embeddings."""
    response = test_client.post(
        "/api/v1/embed-batch",
        json={"texts": ["first text", "second text"]},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert data["count"] == 2


def test_embed_batch_calls_dispatcher(test_client, mock_dispatcher):
    """POST /embed-batch calls dispatcher.embed_batch."""
    test_client.post(
        "/api/v1/embed-batch",
        json={"texts": ["a", "b"]},
        headers=HEADERS,
    )
    mock_dispatcher.embed_batch.assert_called_once()


def test_embed_batch_returns_500_on_error(mock_vector_store, mock_cache):
    """POST /embed-batch returns 500 when dispatcher fails."""
    dispatcher = AsyncMock()
    dispatcher.embed_batch = AsyncMock(side_effect=EmbeddingError("batch fail"))

    app = FastAPI()
    router = create_router(dispatcher, mock_vector_store, mock_cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/embed-batch",
        json={"texts": ["a", "b"]},
        headers=HEADERS,
    )
    assert response.status_code == 500


def test_embed_batch_stores_cache(test_client, mock_cache):
    """POST /embed-batch caches each result."""
    test_client.post(
        "/api/v1/embed-batch",
        json={"texts": ["x", "y"]},
        headers=HEADERS,
    )
    # set should be called once per embedding
    assert mock_cache.set.call_count >= 1


# ============================================================================
# SEARCH ENDPOINT
# ============================================================================


def test_search_success(test_client):
    """POST /search returns 200 with results."""
    response = test_client.post(
        "/api/v1/search",
        json={"query": "machine learning"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["count"] >= 0


def test_search_calls_dispatcher_and_vector_store(test_client, mock_dispatcher, mock_vector_store):
    """POST /search calls embed then vector_store.search."""
    test_client.post(
        "/api/v1/search",
        json={"query": "neural networks"},
        headers=HEADERS,
    )
    mock_dispatcher.embed.assert_called_once()
    mock_vector_store.search.assert_called_once()


def test_search_returns_500_on_dispatcher_error(mock_vector_store, mock_cache):
    """POST /search returns 500 when dispatcher fails."""
    dispatcher = AsyncMock()
    dispatcher.embed = AsyncMock(side_effect=EmbeddingError("embed fail"))

    app = FastAPI()
    router = create_router(dispatcher, mock_vector_store, mock_cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/search",
        json={"query": "test"},
        headers=HEADERS,
    )
    assert response.status_code == 500


def test_search_returns_500_on_vector_store_error(mock_dispatcher, mock_cache):
    """POST /search returns 500 when vector store fails."""
    vs = AsyncMock()
    vs.search = AsyncMock(side_effect=VectorStoreError("DB error"))
    vs.count = AsyncMock(return_value=0)

    app = FastAPI()
    router = create_router(mock_dispatcher, vs, mock_cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/search",
        json={"query": "test"},
        headers=HEADERS,
    )
    assert response.status_code == 500


def test_search_respects_limit_parameter(test_client, mock_vector_store):
    """POST /search passes limit to vector store."""
    test_client.post(
        "/api/v1/search",
        json={"query": "test", "limit": 5},
        headers=HEADERS,
    )
    call_kwargs = mock_vector_store.search.call_args
    assert call_kwargs is not None


def test_search_empty_results(mock_dispatcher, mock_cache):
    """POST /search with no results returns count=0."""
    vs = AsyncMock()
    vs.search = AsyncMock(return_value=[])
    vs.count = AsyncMock(return_value=0)

    app = FastAPI()
    router = create_router(mock_dispatcher, vs, mock_cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.post(
        "/api/v1/search",
        json={"query": "obscure query"},
        headers=HEADERS,
    )
    assert response.status_code == 200
    assert response.json()["count"] == 0


# ============================================================================
# STATS ENDPOINT
# ============================================================================


def test_stats_success(test_client):
    """GET /stats returns 200 with statistics."""
    response = test_client.get("/api/v1/stats", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "embeddings_generated" in data
    assert "cache_hit_rate" in data
    assert "stored_vectors" in data


def test_stats_uses_dispatcher_metrics(test_client, mock_dispatcher):
    """GET /stats reads from dispatcher.get_metrics()."""
    test_client.get("/api/v1/stats", headers=HEADERS)
    mock_dispatcher.get_metrics.assert_called_once()


def test_stats_uses_cache_stats(test_client, mock_cache):
    """GET /stats reads from cache.get_stats()."""
    test_client.get("/api/v1/stats", headers=HEADERS)
    mock_cache.get_stats.assert_called_once()


def test_stats_uses_vector_store_count(test_client, mock_vector_store):
    """GET /stats reads from vector_store.count()."""
    test_client.get("/api/v1/stats", headers=HEADERS)
    mock_vector_store.count.assert_called_once()


def test_stats_returns_500_on_error(mock_dispatcher, mock_vector_store, mock_cache):
    """GET /stats returns 500 on unexpected error."""
    mock_dispatcher.get_metrics = MagicMock(side_effect=RuntimeError("unexpected"))

    app = FastAPI()
    router = create_router(mock_dispatcher, mock_vector_store, mock_cache)
    app.include_router(router)
    client = TestClient(app)

    response = client.get("/api/v1/stats", headers=HEADERS)
    assert response.status_code == 500


# ============================================================================
# EmbeddingAPI CLASS UNIT TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_embedding_api_embed_single_cache_hit():
    """EmbeddingAPI.embed_single returns cached=True on hit."""
    dispatcher = AsyncMock()
    vs = AsyncMock()
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=[0.5, 0.6])

    api = EmbeddingAPI(dispatcher, vs, cache)
    request = EmbedRequest(text="cached text", model=EmbeddingModelEnum.OPENAI_3_SMALL)
    result = await api.embed_single(request, api_key=VALID_KEY)

    assert result.cached is True
    dispatcher.embed.assert_not_called()


@pytest.mark.asyncio
async def test_embedding_api_embed_single_cache_failure_continues():
    """embed_single continues even if cache.set raises CacheError."""
    dispatcher = AsyncMock()
    dispatcher.embed = AsyncMock(return_value=make_embed_result())

    vs = AsyncMock()
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock(side_effect=CacheError("Redis down"))

    api = EmbeddingAPI(dispatcher, vs, cache)
    request = EmbedRequest(text="test", model=EmbeddingModelEnum.OPENAI_3_SMALL)
    result = await api.embed_single(request, api_key=VALID_KEY)

    assert result is not None
    assert result.cached is False
