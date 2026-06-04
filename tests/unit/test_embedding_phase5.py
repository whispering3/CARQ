"""Unit tests for embedding module (Phase 5).

Tests cover:
- Embedding dispatcher (OpenAI integration, rate limiting, retries)
- Vector store (pgvector operations, search)
- Embedding cache (Redis operations, statistics)
"""

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch

import pytest

from carq.embedding.embedding_cache import (
    CacheStats,
    EmbeddingCache,
)
from carq.embedding.embedding_dispatcher import (
    EmbeddingDispatcher,
    EmbeddingModel,
    EmbeddingRequest,
    EmbeddingResult,
)
from carq.embedding.vector_store import (
    EmbeddingRecord,
    SearchResult,
    VectorStore,
    VectorStoreError,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def embedding_request():
    """Create embedding request."""
    return EmbeddingRequest(
        text="This is a test document for embedding.",
        model=EmbeddingModel.OPENAI_3_SMALL,
        chunk_id="test_chunk_001",
    )


@pytest.fixture
def embedding_dispatcher():
    """Create mocked embedding dispatcher."""
    return EmbeddingDispatcher(
        api_key="sk-test-key-12345",
        batch_size=25,
        max_retries=3,
    )


@pytest.fixture
def sample_embedding():
    """Create sample embedding vector."""
    # 1536-dim for ada/3-small
    return [0.1] * 1536


@pytest.fixture
def sample_embeddings():
    """Create multiple sample embeddings."""
    return {
        "text1": [0.1] * 1536,
        "text2": [0.2] * 1536,
        "text3": [0.3] * 1536,
    }


@pytest.fixture
async def embedding_cache():
    """Create embedding cache instance."""
    cache = EmbeddingCache(
        redis_url="redis://localhost:6379",
        ttl_seconds=3600,
    )
    # Don't actually connect in unit tests
    return cache


# ============================================================================
# EMBEDDING DISPATCHER TESTS
# ============================================================================


def test_embedding_dispatcher_initialization(embedding_dispatcher):
    """Test dispatcher initialization."""
    assert embedding_dispatcher.api_key == "sk-test-key-12345"
    assert embedding_dispatcher.batch_size == 25
    assert embedding_dispatcher.max_retries == 3
    assert embedding_dispatcher.embeddings_generated == 0
    assert embedding_dispatcher.total_cost_usd == 0.0


def test_embedding_model_enum():
    """Test embedding model enum."""
    assert EmbeddingModel.OPENAI_3_LARGE.value == "text-embedding-3-large"
    assert EmbeddingModel.OPENAI_3_SMALL.value == "text-embedding-3-small"
    assert EmbeddingModel.OPENAI_ADA.value == "text-embedding-ada-002"


def test_embedding_dispatcher_token_estimation(embedding_dispatcher):
    """Test token count estimation."""
    text = "a" * 400  # ~100 tokens
    tokens = embedding_dispatcher._estimate_tokens(text)
    assert 90 <= tokens <= 110  # Allow margin


def test_embedding_dispatcher_cost_calculation(embedding_dispatcher):
    """Test cost calculation."""
    # 1000 tokens with 3-large ($0.13 per 1M)
    cost = embedding_dispatcher._calculate_cost(1000, EmbeddingModel.OPENAI_3_LARGE)
    expected = (1000 / 1_000_000) * 0.13
    assert abs(cost - expected) < 0.00001


def test_embedding_dispatcher_cost_different_models(embedding_dispatcher):
    """Test cost calculation for different models."""
    tokens = 1000

    cost_large = embedding_dispatcher._calculate_cost(
        tokens,
        EmbeddingModel.OPENAI_3_LARGE,
    )
    cost_small = embedding_dispatcher._calculate_cost(
        tokens,
        EmbeddingModel.OPENAI_3_SMALL,
    )

    # 3-large should be more expensive than 3-small
    assert cost_large > cost_small


def test_embedding_dispatcher_dimension(embedding_dispatcher):
    """Test embedding dimension by model."""
    assert embedding_dispatcher.get_dimension(EmbeddingModel.OPENAI_3_LARGE) == 3072
    assert embedding_dispatcher.get_dimension(EmbeddingModel.OPENAI_3_SMALL) == 1536
    assert embedding_dispatcher.get_dimension(EmbeddingModel.OPENAI_ADA) == 1536


def test_embedding_dispatcher_text_hash(embedding_dispatcher):
    """Test text hashing for caching."""
    text = "Hello, world!"
    hash1 = embedding_dispatcher.text_hash(text)
    hash2 = embedding_dispatcher.text_hash(text)

    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest


def test_embedding_dispatcher_metrics(embedding_dispatcher):
    """Test metrics tracking."""
    metrics = embedding_dispatcher.get_metrics()

    assert metrics["total_embeddings"] == 0
    assert metrics["total_tokens"] == 0
    assert metrics["total_cost_usd"] == 0.0
    assert metrics["avg_cost_per_embedding"] == 0.0


def test_embedding_dispatcher_normalize_embedding():
    """Test embedding normalization helpers."""
    normalized = EmbeddingDispatcher._normalize_embedding([3.0, 4.0])
    assert pytest.approx(normalized[0], rel=1e-6) == 0.6
    assert pytest.approx(normalized[1], rel=1e-6) == 0.8
    assert EmbeddingDispatcher._normalize_embedding([0.0, 0.0]) == [0.0, 0.0]


def test_embedding_request_creation(embedding_request):
    """Test embedding request creation."""
    assert embedding_request.text == "This is a test document for embedding."
    assert embedding_request.model == EmbeddingModel.OPENAI_3_SMALL
    assert embedding_request.chunk_id == "test_chunk_001"


def test_embedding_result_dimension(sample_embedding):
    """Test embedding result dimension property."""
    result = EmbeddingResult(
        text="test",
        embedding=sample_embedding,
        model="text-embedding-3-small",
        tokens_used=10,
    )

    assert result.embedding_dimension == 1536


@pytest.mark.asyncio
async def test_embedding_dispatcher_embed_with_mock(embedding_dispatcher):
    """Test single embedding with mocked API."""
    # Mock the OpenAI API call
    mock_response = {
        "data": [
            {
                "embedding": [0.1] * 1536,
                "index": 0,
            }
        ]
    }

    request = EmbeddingRequest(
        text="Test text",
        model=EmbeddingModel.OPENAI_3_SMALL,
    )

    with patch.object(
        embedding_dispatcher,
        "_call_openai_api_async",
        AsyncMock(return_value=mock_response),
    ):
        result = await embedding_dispatcher.embed(request)

        assert result.text == "Test text"
        assert len(result.embedding) == 1536
        assert result.model == "text-embedding-3-small"
        assert result.tokens_used > 0
        assert result.cost_usd > 0


@pytest.mark.asyncio
async def test_embedding_dispatcher_embed_batch_with_mock(embedding_dispatcher):
    """Test batch embedding with mocked API."""
    requests = [
        EmbeddingRequest(text="one", model=EmbeddingModel.OPENAI_3_SMALL),
        EmbeddingRequest(text="two", model=EmbeddingModel.OPENAI_3_SMALL),
    ]
    mock_response = {
        "data": [
            {"embedding": [0.1] * 1536},
            {"embedding": [0.2] * 1536},
        ]
    }

    with patch.object(embedding_dispatcher.rate_limiter, "acquire", AsyncMock(return_value=True)):
        with patch.object(
            embedding_dispatcher,
            "_call_openai_batch_api_async",
            AsyncMock(return_value=mock_response),
        ):
            results = await embedding_dispatcher.embed_batch(requests)

    assert len(results) == 2
    assert embedding_dispatcher.embeddings_generated == 2
    assert results[0].embedding_dimension == 1536
    assert pytest.approx(sum(value * value for value in results[0].embedding), rel=1e-6) == 1.0


# ============================================================================
# VECTOR STORE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_vector_store_initialization():
    """Test vector store initialization."""
    mock_session = AsyncMock()
    mock_session.add = Mock()
    store = VectorStore(mock_session)

    assert store.session == mock_session


@pytest.mark.asyncio
async def test_vector_store_insert_and_lookup():
    """Test insert, lookup, update, and delete flows."""
    mock_session = Mock()
    mock_session.add = Mock()
    mock_session.flush = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session.delete = AsyncMock()
    mock_session.commit = AsyncMock()
    store = VectorStore(mock_session)
    chunk_id = str(uuid.uuid4())
    embedding_id = uuid.uuid4()
    inserted = Mock(id=embedding_id)

    with patch("carq.embedding.vector_store.Embedding", return_value=inserted):
        inserted_id = await store.insert(
            chunk_id=chunk_id,
            text="hello",
            embedding=[0.1, 0.2],
            model="text-embedding-3-small",
            tokens_used=12,
            cost_usd=0.02,
            metadata={"source": "test"},
        )

    assert inserted_id == str(embedding_id)
    mock_session.add.assert_called_once_with(inserted)

    db_embedding = Mock(
        chunk_id=chunk_id,
        text="hello",
        embedding=[0.1, 0.2],
        model="text-embedding-3-small",
        tokens_used=12,
        cost_usd=0.02,
        attributes={"source": "test"},
    )
    mock_result = Mock()
    mock_result.fetchall.return_value = [
        (uuid.uuid4(), chunk_id, "hello", [0.1, 0.2], 0.9, {"source": "test"})
    ]
    mock_result.scalar_one_or_none.return_value = db_embedding
    mock_session.execute = AsyncMock(return_value=mock_result)

    results = await store.search([0.1, 0.2], limit=5, similarity_threshold=0.1)
    assert len(results) == 1
    assert results[0].chunk_id == chunk_id

    updated = await store.update(chunk_id, [0.5, 0.5], metadata={"updated": True})
    assert updated is True
    assert db_embedding.embedding == [0.5, 0.5]
    assert db_embedding.attributes["updated"] is True

    found = await store.get_by_chunk_id(chunk_id)
    assert found is not None
    assert found.chunk_id == chunk_id

    deleted = await store.delete(chunk_id)
    assert deleted is True
    mock_session.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_vector_store_insert_batch():
    """Test batch insertion path."""
    mock_session = Mock()
    mock_session.add = Mock()
    mock_session.commit = AsyncMock()

    @asynccontextmanager
    async def nested_tx():
        yield None

    mock_session.begin_nested = Mock(return_value=nested_tx())
    store = VectorStore(mock_session)
    records = [
        EmbeddingRecord(
            chunk_id=str(uuid.uuid4()),
            text="one",
            embedding=[0.1, 0.2],
            model="text-embedding-3-small",
            tokens_used=1,
            cost_usd=0.001,
        ),
        EmbeddingRecord(
            chunk_id=str(uuid.uuid4()),
            text="two",
            embedding=[0.2, 0.3],
            model="text-embedding-3-small",
            tokens_used=2,
            cost_usd=0.002,
        ),
    ]

    with patch.object(store, "insert", AsyncMock(side_effect=["id-1", "id-2"])):
        ids = await store.insert_batch(records)

    assert ids == ["id-1", "id-2"]
    mock_session.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_vector_store_insert_batch_failure():
    """Batch insert should surface VectorStoreError."""
    mock_session = Mock()
    mock_session.add = Mock()
    mock_session.commit = AsyncMock()

    @asynccontextmanager
    async def nested_tx():
        yield None

    mock_session.begin_nested = Mock(return_value=nested_tx())
    store = VectorStore(mock_session)
    records = [
        EmbeddingRecord(
            chunk_id=str(uuid.uuid4()),
            text="one",
            embedding=[0.1, 0.2],
            model="text-embedding-3-small",
            tokens_used=1,
            cost_usd=0.001,
        )
    ]
    with patch.object(store, "insert", AsyncMock(side_effect=RuntimeError("db fail"))):
        with pytest.raises(VectorStoreError):
            await store.insert_batch(records)


@pytest.mark.asyncio
async def test_vector_store_not_found_paths():
    """Update/delete/get should handle missing rows."""
    mock_session = Mock()
    mock_session.execute = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.delete = AsyncMock()
    store = VectorStore(mock_session)
    chunk_id = str(uuid.uuid4())

    mock_result = Mock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    assert await store.update(chunk_id, [0.1, 0.2]) is False
    assert await store.delete(chunk_id) is False
    assert await store.get_by_chunk_id(chunk_id) is None


@pytest.mark.asyncio
async def test_vector_store_error_paths():
    """Search/count/delete_all should wrap low-level errors."""
    mock_session = Mock()
    mock_session.execute = AsyncMock(side_effect=RuntimeError("db down"))
    mock_session.commit = AsyncMock()
    store = VectorStore(mock_session)

    with pytest.raises(VectorStoreError):
        await store.search([0.1, 0.2])
    with pytest.raises(VectorStoreError):
        await store.count()
    with pytest.raises(VectorStoreError):
        await store.delete_all()


@pytest.mark.asyncio
async def test_vector_store_count(embedding_dispatcher):
    """Test vector store count operation."""
    mock_session = AsyncMock()
    store = VectorStore(mock_session)

    # Mock count query
    mock_result = Mock()
    mock_result.scalar.return_value = 42

    with patch.object(
        mock_session,
        "execute",
        return_value=mock_result,
    ):
        count = await store.count()
        assert count == 42


@pytest.mark.asyncio
async def test_vector_store_delete_all(embedding_dispatcher):
    """Test delete all embeddings."""
    mock_session = AsyncMock()
    store = VectorStore(mock_session)

    # Mock delete operation
    mock_result = Mock()
    AsyncMock()

    with patch.object(
        mock_session,
        "execute",
        return_value=mock_result,
    ):
        # COUNT query returns 2 (the number of embeddings to be deleted)
        mock_result.scalar.return_value = 2

        count = await store.delete_all()
        assert count == 2


def test_search_result_creation():
    """Test search result creation."""
    result = SearchResult(
        chunk_id="chunk_123",
        text="Document text",
        embedding=[0.1] * 1536,
        similarity_score=0.85,
        metadata={"source": "test"},
    )

    assert result.chunk_id == "chunk_123"
    assert result.similarity_score == 0.85
    assert result.metadata["source"] == "test"


def test_embedding_record_creation():
    """Test embedding record creation."""
    record = EmbeddingRecord(
        chunk_id="chunk_456",
        text="Another document",
        embedding=[0.2] * 1536,
        model="text-embedding-3-small",
        tokens_used=50,
        cost_usd=0.00001,
    )

    assert record.chunk_id == "chunk_456"
    assert record.tokens_used == 50
    assert record.cost_usd == 0.00001


# ============================================================================
# EMBEDDING CACHE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_embedding_cache_initialization(embedding_cache):
    """Test cache initialization."""
    assert embedding_cache.ttl_seconds == 3600
    assert embedding_cache.prefix == "embedding:"
    assert embedding_cache.stats.hits == 0
    assert embedding_cache.stats.misses == 0


def test_cache_stats_hit_rate():
    """Test cache stats hit rate calculation."""
    stats = CacheStats(hits=75, misses=25)
    assert stats.hit_rate == 0.75

    stats_empty = CacheStats()
    assert stats_empty.hit_rate == 0.0


def test_cache_stats_string_representation():
    """Test cache stats string representation."""
    stats = CacheStats(hits=100, misses=50, size=1000)
    stats_str = str(stats)

    assert "hits=100" in stats_str
    assert "misses=50" in stats_str
    assert "size=1000" in stats_str
    assert "66.67%" in stats_str  # hit_rate


def test_embedding_cache_key_generation(embedding_cache):
    """Test cache key generation."""
    text = "Hello, world!"
    key1 = embedding_cache._make_key(text)
    key2 = embedding_cache._make_key(text)

    assert key1 == key2
    assert key1.startswith("embedding:")
    assert len(key1) == len("embedding:") + 64  # SHA-256


def test_embedding_cache_different_texts(embedding_cache):
    """Test different texts produce different keys."""
    text1 = "First text"
    text2 = "Second text"

    key1 = embedding_cache._make_key(text1)
    key2 = embedding_cache._make_key(text2)

    assert key1 != key2


@pytest.mark.asyncio
async def test_embedding_cache_get_no_client(embedding_cache):
    """Test get operation without Redis client."""
    embedding_cache.client = None
    result = await embedding_cache.get("test text")
    assert result is None


@pytest.mark.asyncio
async def test_embedding_cache_set_no_client(embedding_cache):
    """Test set operation without Redis client."""
    embedding_cache.client = None
    result = await embedding_cache.set("test text", [0.1] * 1536)
    assert result is False


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_embedding_dispatcher_metrics_update():
    """Test metrics update after embedding generation."""
    dispatcher = EmbeddingDispatcher(api_key="sk-test")

    # Simulate embedding
    request = EmbeddingRequest(text="Test")
    mock_response = {"data": [{"embedding": [0.1] * 1536}]}

    with patch.object(dispatcher, "_call_openai_api_async", AsyncMock(return_value=mock_response)):
        await dispatcher.embed(request)

        metrics = dispatcher.get_metrics()
        assert metrics["total_embeddings"] == 1
        assert metrics["total_tokens"] > 0
        assert metrics["total_cost_usd"] > 0


def test_embedding_request_batch():
    """Test creating multiple embedding requests."""
    texts = [f"Text {i}" for i in range(5)]
    requests = [EmbeddingRequest(text=text) for text in texts]

    assert len(requests) == 5
    assert all(isinstance(r, EmbeddingRequest) for r in requests)


def test_embedding_models_pricing():
    """Test embedding model pricing information."""
    prices = EmbeddingDispatcher.PRICING

    assert EmbeddingModel.OPENAI_3_LARGE in prices
    assert EmbeddingModel.OPENAI_3_SMALL in prices
    assert EmbeddingModel.OPENAI_ADA in prices

    # 3-large should be more expensive than 3-small
    assert prices[EmbeddingModel.OPENAI_3_LARGE] > prices[EmbeddingModel.OPENAI_3_SMALL]


def test_embedding_models_dimensions():
    """Test embedding model dimensions."""
    dims = EmbeddingDispatcher.DIMENSIONS

    assert dims[EmbeddingModel.OPENAI_3_LARGE] == 3072
    assert dims[EmbeddingModel.OPENAI_3_SMALL] == 1536
    assert dims[EmbeddingModel.OPENAI_ADA] == 1536
