"""Cache de embeddings com backend Redis e degradação graciosa."""

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

import redis.asyncio as redis

from carq.core.exceptions import ProcessingError

logger = logging.getLogger(__name__)


class CacheError(ProcessingError):
    """Levantada quando operações de cache falham."""

    pass


@dataclass
class CacheStats:
    """Estatísticas do cache."""

    hits: int = 0
    misses: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total

    def __str__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"size={self.size}, hit_rate={self.hit_rate:.2%})"
        )


class EmbeddingCache:
    """Cache de embeddings com backend Redis, TTL e degradação graciosa."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 86400 * 30,  # 30 dias
        prefix: str = "embedding:",
    ):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)

        self.client: Optional[redis.Redis] = None
        self.stats = CacheStats()

    async def connect(self):
        """Conecta ao Redis."""
        try:
            self.client = redis.from_url(self.redis_url)
            await self.client.ping()
            self.logger.info(
                "Connected to Redis cache",
                extra={"url": self.redis_url},
            )
        except Exception as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise CacheError(f"Redis connection failed: {e}") from e

    async def disconnect(self):
        """Desconecta do Redis."""
        if self.client:
            await self.client.close()
            self.logger.info("Disconnected from Redis cache")

    async def get(self, text: str) -> Optional[list[float]]:
        """Busca embedding em cache. Retorna None em miss ou sem conexão."""
        if not self.client:
            return None

        try:
            key = self._make_key(text)
            value = await self.client.get(key)

            if value:
                embedding = json.loads(value)
                self.stats.hits += 1
                self.logger.debug(
                    "Cache hit for text",
                    extra={"key": key},
                )
                return embedding
            else:
                self.stats.misses += 1
                self.logger.debug(
                    "Cache miss for text",
                    extra={"key": key},
                )
                return None

        except json.JSONDecodeError as e:
            self.logger.warning(f"Invalid cached embedding: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Cache get failed: {e}")
            raise CacheError(f"Cache get failed: {e}") from e

    async def set(self, text: str, embedding: list[float]) -> bool:
        """Armazena embedding em cache. Retorna False sem conexão."""
        if not self.client:
            return False

        try:
            key = self._make_key(text)
            value = json.dumps(embedding)

            await self.client.setex(
                key,
                self.ttl_seconds,
                value,
            )

            self.logger.debug(
                "Cached embedding for text",
                extra={"key": key, "ttl": self.ttl_seconds},
            )

            return True

        except Exception as e:
            self.logger.error(f"Cache set failed: {e}")
            raise CacheError(f"Cache set failed: {e}") from e

    async def get_batch(self, texts: list[str]) -> dict[str, Optional[list[float]]]:
        """Busca múltiplos embeddings em cache usando MGET do Redis (operação única)."""
        if not self.client or not texts:
            return dict.fromkeys(texts)

        try:
            keys = [self._make_key(t) for t in texts]
            values = await self.client.mget(*keys)

            results: dict[str, Optional[list[float]]] = {}
            for text, value in zip(texts, values, strict=False):
                if value is not None:
                    try:
                        results[text] = json.loads(value)
                        self.stats.hits += 1
                    except json.JSONDecodeError:
                        self.logger.warning("Invalid cached embedding in batch", extra={"key": self._make_key(text)})
                        results[text] = None
                        self.stats.misses += 1
                else:
                    results[text] = None
                    self.stats.misses += 1

            return results

        except Exception as e:
            self.logger.error(f"Batch get (MGET) falhou: {e}")
            # Degradação graciosa: retorna None para todos
            return dict.fromkeys(texts)

    async def set_batch(
        self,
        embeddings: dict[str, list[float]],
    ) -> int:
        """Armazena múltiplos embeddings em cache. Retorna a quantidade armazenada."""
        cached_count = 0

        for text, embedding in embeddings.items():
            try:
                if await self.set(text, embedding):
                    cached_count += 1
            except Exception as e:
                self.logger.warning(f"Batch set error for text: {e}")

        return cached_count

    async def delete(self, text: str) -> bool:
        """Remove embedding do cache. Retorna True se removido."""
        if not self.client:
            return False

        try:
            key = self._make_key(text)
            deleted = await self.client.delete(key)

            if deleted:
                self.logger.debug("Deleted cached embedding", extra={"key": key})

            return bool(deleted)

        except Exception as e:
            self.logger.error(f"Cache delete failed: {e}")
            raise CacheError(f"Cache delete failed: {e}") from e

    async def clear(self) -> int:
        """Remove todos os embeddings em cache. Retorna número de chaves removidas."""
        if not self.client:
            return 0

        try:
            pattern = f"{self.prefix}*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern)
                if keys:
                    deleted += await self.client.delete(*keys)
                if cursor == 0:
                    break

            self.logger.warning(
                f"Cleared {deleted} cached embeddings",
                extra={"count": deleted},
            )

            return deleted

        except Exception as e:
            self.logger.error(f"Cache clear failed: {e}")
            raise CacheError(f"Cache clear failed: {e}") from e

    async def get_stats(self) -> CacheStats:
        if not self.client:
            return self.stats

        try:
            pattern = f"{self.prefix}*"
            cursor = 0
            size = 0

            while True:
                cursor, keys = await self.client.scan(cursor, match=pattern)
                size += len(keys)
                if cursor == 0:
                    break

            self.stats.size = size
            return self.stats

        except Exception as e:
            self.logger.error(f"Stats retrieval failed: {e}")
            return self.stats

    async def reset_stats(self):
        self.stats = CacheStats()

    def _make_key(self, text: str) -> str:
        """Gera chave de cache usando SHA-256 do texto para chaves consistentes e curtas."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{self.prefix}{text_hash}"


async def get_or_embed(
    cache: EmbeddingCache,
    dispatcher,
    text: str,
) -> tuple[list[float], bool]:
    """Busca embedding do cache ou gera um novo (padrão cache-aside)."""
    cached = await cache.get(text)
    if cached is not None:
        return cached, True

    from carq.embedding.embedding_dispatcher import EmbeddingRequest

    request = EmbeddingRequest(text=text)
    result = await dispatcher.embed(request)

    await cache.set(text, result.embedding)

    return result.embedding, False
