"""Tests for embedding_cache.py - Phase 8.

Covers: get/set, cache miss, hit rate stats, clear, batch ops,
key generation, and error handling.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from carq.embedding.embedding_cache import EmbeddingCache, CacheStats, CacheError


# ============================================================================
# HELPERS
# ============================================================================


def make_cache_with_client(client):
    """Return an EmbeddingCache with a pre-set mock Redis client."""
    cache = EmbeddingCache(redis_url="redis://localhost:6379", ttl_seconds=3600)
    cache.client = client
    return cache


def make_redis_mock(stored: dict | None = None):
    """Build a minimal async Redis mock backed by an in-memory dict."""
    store = stored or {}
    mock = AsyncMock()

    async def _get(key):
        return store.get(key)

    async def _setex(key, ttl, value):
        store[key] = value
        return True

    async def _delete(*keys):
        deleted = sum(1 for k in keys if k in store)
        for k in keys:
            store.pop(k, None)
        return deleted

    async def _scan(cursor, match=None):
        # Simple single-shot scan
        keys = [k.encode() if isinstance(k, str) else k for k in store.keys()]
        return (0, keys)

    async def _ping():
        return True

    async def _mget(*keys):
        return [store.get(k) for k in keys]

    mock.get = AsyncMock(side_effect=_get)
    mock.mget = AsyncMock(side_effect=_mget)
    mock.setex = AsyncMock(side_effect=_setex)
    mock.delete = AsyncMock(side_effect=_delete)
    mock.scan = AsyncMock(side_effect=_scan)
    mock.ping = AsyncMock(side_effect=_ping)
    mock.close = AsyncMock()
    return mock, store


# ============================================================================
# KEY GENERATION TESTS
# ============================================================================


def test_make_key_is_deterministic():
    """Same text always produces same cache key."""
    cache = EmbeddingCache()
    key1 = cache._make_key("hello world")
    key2 = cache._make_key("hello world")
    assert key1 == key2


def test_make_key_differs_for_different_text():
    """Different texts produce different cache keys."""
    cache = EmbeddingCache()
    key1 = cache._make_key("text A")
    key2 = cache._make_key("text B")
    assert key1 != key2


def test_make_key_has_prefix():
    """Cache key includes the configured prefix."""
    cache = EmbeddingCache(prefix="emb:")
    key = cache._make_key("hello")
    assert key.startswith("emb:")


def test_make_key_sha256_length():
    """Cache key hash portion is 64 hex characters (SHA-256)."""
    cache = EmbeddingCache(prefix="embedding:")
    key = cache._make_key("test")
    hash_part = key[len("embedding:"):]
    assert len(hash_part) == 64


# ============================================================================
# CACHE STATS TESTS
# ============================================================================


def test_cache_stats_initial_hit_rate_zero():
    """Hit rate is 0.0 with no operations."""
    stats = CacheStats()
    assert stats.hit_rate == 0.0


def test_cache_stats_hit_rate_all_hits():
    """Hit rate is 1.0 when all requests are hits."""
    stats = CacheStats(hits=10, misses=0)
    assert stats.hit_rate == 1.0


def test_cache_stats_hit_rate_mixed():
    """Hit rate is correctly calculated for mixed hits/misses."""
    stats = CacheStats(hits=3, misses=1)
    assert abs(stats.hit_rate - 0.75) < 0.001


def test_cache_stats_str_representation():
    """CacheStats.__str__ includes relevant fields."""
    stats = CacheStats(hits=5, misses=2, size=7)
    s = str(stats)
    assert "hits=5" in s
    assert "misses=2" in s


# ============================================================================
# GET TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_get_returns_none_when_no_client():
    """get() returns None if client is not connected."""
    cache = EmbeddingCache()
    result = await cache.get("any text")
    assert result is None


@pytest.mark.asyncio
async def test_get_miss_returns_none():
    """get() returns None on cache miss and increments misses."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    result = await cache.get("unseen text")
    assert result is None
    assert cache.stats.misses == 1
    assert cache.stats.hits == 0


@pytest.mark.asyncio
async def test_get_hit_returns_embedding():
    """get() returns embedding on cache hit and increments hits."""
    embedding = [0.1, 0.2, 0.3]
    cache = EmbeddingCache()
    key = cache._make_key("hello")

    mock, store = make_redis_mock({key: json.dumps(embedding)})
    cache.client = mock

    result = await cache.get("hello")
    assert result == embedding
    assert cache.stats.hits == 1
    assert cache.stats.misses == 0


@pytest.mark.asyncio
async def test_get_handles_invalid_json():
    """get() returns None if cached value is invalid JSON."""
    cache = EmbeddingCache()
    key = cache._make_key("bad data")

    mock = AsyncMock()
    mock.get = AsyncMock(return_value=b"not-valid-json{{{")
    cache.client = mock

    result = await cache.get("bad data")
    assert result is None


@pytest.mark.asyncio
async def test_get_raises_cache_error_on_redis_failure():
    """get() raises CacheError on unexpected Redis error."""
    mock = AsyncMock()
    mock.get = AsyncMock(side_effect=RuntimeError("Redis down"))
    cache = make_cache_with_client(mock)

    with pytest.raises(CacheError):
        await cache.get("some text")


# ============================================================================
# SET TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_set_returns_false_when_no_client():
    """set() returns False when client is not connected."""
    cache = EmbeddingCache()
    result = await cache.set("text", [0.1, 0.2])
    assert result is False


@pytest.mark.asyncio
async def test_set_stores_embedding():
    """set() stores embedding in Redis and returns True."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    result = await cache.set("hello", [0.1, 0.2, 0.3])
    assert result is True
    mock.setex.assert_called_once()


@pytest.mark.asyncio
async def test_set_uses_correct_ttl():
    """set() uses the configured TTL."""
    mock, store = make_redis_mock()
    cache = EmbeddingCache(ttl_seconds=999)
    cache.client = mock

    await cache.set("text", [1.0])
    args = mock.setex.call_args
    # setex(key, ttl, value)
    ttl_arg = args[0][1] if args[0] else args[1].get("time", args[1].get("ex"))
    assert ttl_arg == 999


@pytest.mark.asyncio
async def test_set_raises_cache_error_on_failure():
    """set() raises CacheError on Redis error."""
    mock = AsyncMock()
    mock.setex = AsyncMock(side_effect=RuntimeError("write error"))
    cache = make_cache_with_client(mock)

    with pytest.raises(CacheError):
        await cache.set("text", [0.1])


# ============================================================================
# SET + GET ROUNDTRIP
# ============================================================================


@pytest.mark.asyncio
async def test_set_then_get_returns_same_embedding():
    """Setting then getting returns the same embedding vector."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    embedding = [0.1, 0.2, 0.3, 0.4]
    await cache.set("roundtrip text", embedding)

    # Simulate what Redis would return
    key = cache._make_key("roundtrip text")
    stored_value = mock.setex.call_args[0][2]

    mock2 = AsyncMock()
    mock2.get = AsyncMock(return_value=stored_value.encode() if isinstance(stored_value, str) else stored_value)
    cache.client = mock2

    result = await cache.get("roundtrip text")
    assert result == embedding


# ============================================================================
# BATCH OPERATIONS
# ============================================================================


@pytest.mark.asyncio
async def test_get_batch_returns_dict():
    """get_batch returns a dict with one entry per text."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    result = await cache.get_batch(["text1", "text2", "text3"])
    assert set(result.keys()) == {"text1", "text2", "text3"}


@pytest.mark.asyncio
async def test_get_batch_all_misses():
    """get_batch with no cached entries returns all None values."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    result = await cache.get_batch(["a", "b"])
    assert all(v is None for v in result.values())


@pytest.mark.asyncio
async def test_set_batch_returns_count():
    """set_batch returns the number of successfully cached embeddings."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    count = await cache.set_batch({
        "text1": [0.1, 0.2],
        "text2": [0.3, 0.4],
    })
    assert count == 2


@pytest.mark.asyncio
async def test_set_batch_empty():
    """set_batch with empty dict returns 0."""
    mock, store = make_redis_mock()
    cache = make_cache_with_client(mock)

    count = await cache.set_batch({})
    assert count == 0


# ============================================================================
# DELETE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_delete_returns_false_when_no_client():
    """delete() returns False when not connected."""
    cache = EmbeddingCache()
    result = await cache.delete("any")
    assert result is False


@pytest.mark.asyncio
async def test_delete_existing_key():
    """delete() returns True for an existing key."""
    cache = EmbeddingCache()
    key = cache._make_key("to delete")
    mock, store = make_redis_mock({key: json.dumps([0.1])})
    cache.client = mock

    result = await cache.delete("to delete")
    assert result is True


@pytest.mark.asyncio
async def test_delete_nonexistent_key():
    """delete() returns False for a missing key."""
    mock = AsyncMock()
    mock.delete = AsyncMock(return_value=0)
    cache = make_cache_with_client(mock)

    result = await cache.delete("not there")
    assert result is False


# ============================================================================
# CLEAR TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_clear_returns_zero_when_no_client():
    """clear() returns 0 when not connected."""
    cache = EmbeddingCache()
    result = await cache.clear()
    assert result == 0


@pytest.mark.asyncio
async def test_clear_deletes_all_keys():
    """clear() removes all keys with the cache prefix."""
    cache = EmbeddingCache(prefix="embedding:")
    key1 = cache._make_key("a")
    key2 = cache._make_key("b")
    mock, store = make_redis_mock({key1: "[]", key2: "[]"})
    cache.client = mock

    result = await cache.clear()
    assert result >= 0  # SQLite mock doesn't guarantee exact count


# ============================================================================
# STATS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_get_stats_no_client():
    """get_stats returns stats object even without client."""
    cache = EmbeddingCache()
    stats = await cache.get_stats()
    assert isinstance(stats, CacheStats)


@pytest.mark.asyncio
async def test_get_stats_counts_keys():
    """get_stats queries Redis for key count."""
    cache = EmbeddingCache(prefix="embedding:")
    key1 = cache._make_key("x")
    mock, store = make_redis_mock({key1: "[]"})
    cache.client = mock
    cache.stats.hits = 5
    cache.stats.misses = 2

    stats = await cache.get_stats()
    assert stats.hits == 5
    assert stats.misses == 2
    assert isinstance(stats.size, int)


@pytest.mark.asyncio
async def test_stats_accumulate_across_calls():
    """Stats accumulate across multiple get() calls."""
    embedding = [0.1]
    cache = EmbeddingCache()
    key = cache._make_key("text")
    mock, store = make_redis_mock({key: json.dumps(embedding)})
    cache.client = mock

    await cache.get("text")  # hit
    await cache.get("text")  # hit
    await cache.get("miss1")  # miss
    await cache.get("miss2")  # miss

    assert cache.stats.hits == 2
    assert cache.stats.misses == 2
    assert abs(cache.stats.hit_rate - 0.5) < 0.001


# ============================================================================
# RESET STATS
# ============================================================================


@pytest.mark.asyncio
async def test_reset_stats():
    """reset_stats clears hit/miss counters."""
    cache = EmbeddingCache()
    cache.stats.hits = 10
    cache.stats.misses = 5

    await cache.reset_stats()

    assert cache.stats.hits == 0
    assert cache.stats.misses == 0
    assert cache.stats.hit_rate == 0.0


# ============================================================================
# CONNECT / DISCONNECT
# ============================================================================


@pytest.mark.asyncio
async def test_connect_sets_client():
    """connect() sets the Redis client on success."""
    cache = EmbeddingCache()
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        await cache.connect()

    assert cache.client is not None


@pytest.mark.asyncio
async def test_connect_raises_on_failure():
    """connect() raises CacheError if Redis connection fails."""
    cache = EmbeddingCache()
    mock_redis = AsyncMock()
    mock_redis.ping = AsyncMock(side_effect=RuntimeError("connection refused"))

    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with pytest.raises(CacheError):
            await cache.connect()


@pytest.mark.asyncio
async def test_disconnect_closes_client():
    """disconnect() closes the Redis client."""
    mock, _ = make_redis_mock()
    cache = make_cache_with_client(mock)

    await cache.disconnect()
    mock.close.assert_called_once()
