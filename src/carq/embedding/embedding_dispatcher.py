"""Dispatcher de embeddings com suporte a OpenAI, limitação de taxa e normalização de vetores."""

import hashlib
import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from carq.core.exceptions import ProcessingError
from carq.queue.backoff_strategy import BackoffConfig, ExponentialBackoff, RetryPolicy
from carq.queue.rate_limiter import RateLimitConfig, RateLimiter, RateLimitProvider

logger = logging.getLogger(__name__)


class EmbeddingModel(str, Enum):
    """Modelos de embedding suportados."""

    OPENAI_3_LARGE = "text-embedding-3-large"  # 3072 dims
    OPENAI_3_SMALL = "text-embedding-3-small"  # 1536 dims
    OPENAI_ADA = "text-embedding-ada-002"  # 1536 dims


class EmbeddingError(ProcessingError):
    """Levantada quando a geração de embeddings falha."""

    pass


@dataclass
class EmbeddingRequest:
    """Requisição para geração de embedding."""

    text: str
    model: EmbeddingModel = EmbeddingModel.OPENAI_3_LARGE
    chunk_id: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddingResult:
    """Resultado da geração de embedding."""

    text: str
    embedding: list[float]
    model: str
    tokens_used: int = 0
    chunk_id: Optional[str] = None
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def embedding_dimension(self) -> int:
        """Retorna a dimensão do embedding."""
        return len(self.embedding)


class EmbeddingDispatcher:
    """Gerencia requisições de embedding com limitação de taxa e retries."""

    # Preço por 1M tokens de entrada (em 2024)
    PRICING = {
        EmbeddingModel.OPENAI_3_LARGE: 0.13,  # $0,13/1M tokens
        EmbeddingModel.OPENAI_3_SMALL: 0.02,  # $0,02/1M tokens
        EmbeddingModel.OPENAI_ADA: 0.10,  # $0,10/1M tokens
    }

    DIMENSIONS = {
        EmbeddingModel.OPENAI_3_LARGE: 3072,
        EmbeddingModel.OPENAI_3_SMALL: 1536,
        EmbeddingModel.OPENAI_ADA: 1536,
    }

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        batch_size: int = 25,
        max_retries: int = 3,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.batch_size = min(batch_size, 2048)
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

        # Cliente assíncrono OpenAI (API v1.0.0+)
        from openai import AsyncOpenAI
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._openai_client = AsyncOpenAI(**kwargs)

        self.rate_limiter = rate_limiter or RateLimiter()
        self._setup_rate_limits()

        backoff_config = BackoffConfig(
            initial_delay=1.0,
            max_delay=60.0,
            exponential_base=2.0,
            jitter_factor=0.1,
        )
        self.retry_policy = RetryPolicy(
            max_attempts=max_retries,
            backoff=ExponentialBackoff(backoff_config),
        )

        self.total_tokens_used = 0
        self.total_cost_usd = 0.0
        self.embeddings_generated = 0

    def _setup_rate_limits(self):
        """Configura os limites de taxa da OpenAI."""
        # Limites de taxa da OpenAI (por nível)
        self.rate_limiter.register_provider(
            RateLimitConfig(
                provider=RateLimitProvider.OPENAI,
                rpm=3000,  # Requisições por minuto
                tpm=1500000,  # Tokens por minuto
                burst_capacity=1000.0,
            )
        )

    async def embed(
        self,
        request: EmbeddingRequest,
        use_cache: bool = True,
    ) -> EmbeddingResult:
        """Gera embedding para um único texto."""
        return await self.retry_policy.execute_with_retry(
            lambda: self._embed_internal(request)
        )

    async def embed_batch(
        self,
        requests: list[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        """Gera embeddings para um lote de textos com limitação de taxa automática."""
        results = []
        total_tokens = sum(self._estimate_tokens(r.text) for r in requests)

        # Verifica capacidade do limitador de taxa
        self.logger.info(
            f"Embedding batch of {len(requests)} texts ({total_tokens} tokens)",
            extra={"batch_size": len(requests), "tokens": total_tokens},
        )

        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]

            batch_tokens = sum(self._estimate_tokens(r.text) for r in batch)
            await self.rate_limiter.acquire(
                RateLimitProvider.OPENAI,
                request_tokens=batch_tokens,
                wait=True,
            )

            batch_results = await self._embed_batch_internal(batch)
            results.extend(batch_results)

        return results

    async def _embed_internal(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingResult:
        """Geração interna de embedding com tratamento de erros."""
        try:
            tokens = self._estimate_tokens(request.text)

            await self.rate_limiter.acquire(
                RateLimitProvider.OPENAI,
                request_tokens=tokens,
                wait=True,
            )

            # Chama a API OpenAI de forma assíncrona (sem overhead de thread)
            response = await self._call_openai_api_async(
                request.text,
                request.model.value,
            )

            embedding = response["data"][0]["embedding"]

            # CORREÇÃO CRÍTICA: Normaliza o embedding para similaridade cosseno consistente
            embedding = self._normalize_embedding(embedding)

            cost = self._calculate_cost(tokens, request.model)

            self.total_tokens_used += tokens
            self.total_cost_usd += cost
            self.embeddings_generated += 1

            self.logger.debug(
                f"Generated embedding for {request.chunk_id or 'unknown'}",
                extra={
                    "chunk_id": request.chunk_id,
                    "model": request.model.value,
                    "tokens": tokens,
                    "cost_usd": cost,
                },
            )

            return EmbeddingResult(
                text=request.text,
                embedding=embedding,
                model=request.model.value,
                tokens_used=tokens,
                chunk_id=request.chunk_id,
                cost_usd=cost,
                cached=False,
            )

        except Exception as e:
            self.logger.error(
                f"Embedding generation failed: {e}",
                extra={
                    "chunk_id": request.chunk_id,
                    "error_type": type(e).__name__,
                },
            )
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    async def _embed_batch_internal(
        self,
        requests: list[EmbeddingRequest],
    ) -> list[EmbeddingResult]:
        """Geração interna de embeddings em lote."""
        try:
            texts = [r.text for r in requests]

            # Chama a API OpenAI de forma assíncrona (sem overhead de thread)
            response = await self._call_openai_batch_api_async(
                texts,
                requests[0].model.value,
            )

            results = []
            for i, embedding_data in enumerate(response["data"]):
                req = requests[i]
                tokens = self._estimate_tokens(req.text)
                cost = self._calculate_cost(tokens, req.model)
                embedding = self._normalize_embedding(embedding_data["embedding"])

                self.total_tokens_used += tokens
                self.total_cost_usd += cost
                self.embeddings_generated += 1

                results.append(
                    EmbeddingResult(
                        text=req.text,
                        embedding=embedding,
                        model=req.model.value,
                        tokens_used=tokens,
                        chunk_id=req.chunk_id,
                        cost_usd=cost,
                        cached=False,
                    )
                )

            return results

        except Exception as e:
            self.logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Failed to generate batch embeddings: {e}") from e

    def _call_openai_api(
        self,
        text: str,
        model: str,
    ) -> dict:
        """Chama a API OpenAI de forma síncrona (openai>=1.0.0). Mantido para compatibilidade com testes."""
        import openai as _openai
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = _openai.OpenAI(**kwargs)
        response = client.embeddings.create(input=text, model=model)
        return {
            "data": [{"embedding": item.embedding} for item in response.data],
            "usage": {"total_tokens": response.usage.total_tokens},
        }

    async def _call_openai_api_async(
        self,
        text: str,
        model: str,
    ) -> dict:
        """Chama a API OpenAI de forma assíncrona (openai>=1.0.0)."""
        response = await self._openai_client.embeddings.create(
            input=text,
            model=model,
        )
        return {
            "data": [{"embedding": item.embedding} for item in response.data],
            "usage": {"total_tokens": response.usage.total_tokens},
        }

    def _call_openai_batch_api(
        self,
        texts: list[str],
        model: str,
    ) -> dict:
        """Chama a API OpenAI para um lote de textos de forma síncrona. Mantido para compatibilidade com testes."""
        import openai as _openai
        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        client = _openai.OpenAI(**kwargs)
        response = client.embeddings.create(input=texts, model=model)
        return {
            "data": [{"embedding": item.embedding} for item in response.data],
            "usage": {"total_tokens": response.usage.total_tokens},
        }

    async def _call_openai_batch_api_async(
        self,
        texts: list[str],
        model: str,
    ) -> dict:
        """Chama a API OpenAI para um lote de textos de forma assíncrona (openai>=1.0.0)."""
        response = await self._openai_client.embeddings.create(
            input=texts,
            model=model,
        )
        return {
            "data": [{"embedding": item.embedding} for item in response.data],
            "usage": {"total_tokens": response.usage.total_tokens},
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estima a quantidade de tokens.

        Aproximação grosseira: 1 token ≈ 4 caracteres.
        """
        return max(1, len(text) // 4)

    @staticmethod
    def _normalize_embedding(embedding: list[float]) -> list[float]:
        """Normaliza o vetor para comprimento unitário (norma L2 = 1).

        CORREÇÃO CRÍTICA: Garante que a similaridade cosseno do pgvector esteja no intervalo [0, 1].
        """
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm == 0:
            return embedding
        return [x / norm for x in embedding]

    @staticmethod
    def _calculate_cost(tokens: int, model: EmbeddingModel) -> float:
        """Calcula o custo em dólares.

        Fórmula: (tokens / 1.000.000) * preço_por_1m_tokens
        """
        price = EmbeddingDispatcher.PRICING.get(model, 0.10)
        return (tokens / 1_000_000) * price

    def get_dimension(self, model: EmbeddingModel) -> int:
        return self.DIMENSIONS.get(model, 1536)

    def get_metrics(self) -> dict:
        return {
            "total_embeddings": self.embeddings_generated,
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 8),
            "avg_cost_per_embedding": (
                round(self.total_cost_usd / self.embeddings_generated, 6)
                if self.embeddings_generated > 0
                else 0.0
            ),
        }

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
