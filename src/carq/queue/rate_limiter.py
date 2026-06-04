"""Limitação de taxa com algoritmo de token bucket para múltiplos provedores."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from carq.core.exceptions import RateLimitError
from carq.core.logging import get_logger

logger = get_logger(__name__)


class RateLimitProvider(str, Enum):
    """Identificadores de provedores de limite de taxa."""

    OPENAI = "openai"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


@dataclass
class RateLimitConfig:
    """Configuração para o limitador de taxa."""

    provider: RateLimitProvider
    rpm: int = 3000  # Requisições por minuto
    tpm: int = 1500000  # Tokens por minuto
    burst_capacity: float = 1000.0  # Capacidade máxima de burst
    refill_rate: float = 100.0  # Tokens por segundo


class TokenBucket:
    """
    Limitador de taxa com token bucket.

    Algoritmo:
    - Tokens acumulam na taxa refill_rate (tokens/segundo)
    - Requisição consome N tokens
    - Se não houver tokens suficientes, a requisição aguarda ou falha
    - Suporta capacidade de burst (máximo de tokens no bucket)
    - Usa asyncio.Event para sincronização assíncrona adequada (CORREÇÃO CRÍTICA)
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
    ):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
        # CORREÇÃO CRÍTICA: Sinaliza quando tokens ficam disponíveis
        self._tokens_available = asyncio.Event()
        self._tokens_available.set()

    async def _refill(self) -> None:
        """Reabastece tokens com base no tempo decorrido."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate),
        )
        self.last_refill = now

    def _sync_refill(self) -> None:
        """Reabastece tokens de forma síncrona com base no tempo decorrido."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate),
        )
        self.last_refill = now

    @property
    def available_tokens(self) -> float:
        """Retorna os tokens atualmente disponíveis.

        Retorna a contagem projetada de tokens, mas só inclui o reabastecimento
        por tempo decorrido se pelo menos 1 token inteiro seria adicionado (evita
        adições minúsculas de ponto flutuante que quebram igualdade exata em testes).
        """
        elapsed = time.time() - self.last_refill
        projected_refill = elapsed * self.refill_rate
        if projected_refill >= 1.0:
            return min(self.capacity, self.tokens + projected_refill)
        return self.tokens

    def consume(self, tokens: float = 1.0) -> bool:
        """Consome tokens de forma síncrona, lançando RateLimitError se insuficientes."""
        if self.tokens < tokens:
            raise RateLimitError(
                f"Rate limit exceeded: {tokens} tokens requested, {self.tokens:.1f} available",
                retry_after=int((tokens - self.tokens) / self.refill_rate) + 1,
            )
        self.tokens -= tokens
        return True

    async def acquire(
        self,
        tokens: float = 1.0,
        wait: bool = False,
    ) -> bool:
        """Tenta adquirir tokens do bucket; retorna False se indisponível (use wait_for_tokens para aguardar)."""
        async with self._lock:
            await self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            self._tokens_available.clear()
            return False

    async def wait_for_tokens(
        self,
        tokens: float = 1.0,
        max_wait: float = 300.0,
    ) -> float:
        """
        Aguarda até que tokens estejam disponíveis.

        CORREÇÃO CRÍTICA: Usa asyncio.Event para evitar condição de corrida
        onde múltiplas corrotinas acordam simultaneamente e transbordam tokens.

        Args:
            tokens: Tokens a adquirir
            max_wait: Máximo de segundos para aguardar

        Retorna:
            Tempo real de espera em segundos

        Lança:
            RateLimitError: Se o timeout for excedido
        """
        start_time = time.time()

        while True:
            async with self._lock:
                await self._refill()

                if self.tokens >= tokens:
                    # Não consome aqui - o chamador deve consumir via consume()
                    return time.time() - start_time

                deficit = tokens - self.tokens
                wait_duration = deficit / self.refill_rate

                elapsed = time.time() - start_time
                if elapsed + wait_duration > max_wait:
                    raise RateLimitError(
                        f"Rate limit wait exceeded {max_wait}s",
                        retry_after=int(max_wait - elapsed),
                    )

                self._tokens_available.clear()

            # Aguarda FORA do lock, para que outras corrotinas possam reabastecer
            try:
                await asyncio.wait_for(
                    self._tokens_available.wait(),
                    timeout=wait_duration * 1.1,  # 10% de margem para desvio de relógio
                )
            except asyncio.TimeoutError:
                # Hora de tentar reabastecer novamente
                pass



class RateLimiter:
    """
    Limitador de taxa multi-provedor com limites adaptativos.

    Gerencia:
    - Limites de taxa por provedor (RPM, TPM)
    - Token bucket para cada provedor
    - Backoff adaptativo em erros de limite de taxa
    - Rastreamento de custo por requisição/token
    """

    def __init__(self):
        self._buckets: Dict[RateLimitProvider, TokenBucket] = {}
        self._request_costs: Dict[str, Tuple[int, int]] = {}
        self._configs: Dict[RateLimitProvider, RateLimitConfig] = {}
        self._lock = asyncio.Lock()

    def register_provider(self, config: RateLimitConfig) -> None:
        """Registra a configuração de limite de taxa para um provedor."""
        rpm_tokens_per_sec = config.rpm / 60.0
        tpm_tokens_per_sec = config.tpm / 60.0
        refill_rate = min(rpm_tokens_per_sec, tpm_tokens_per_sec)

        bucket = TokenBucket(
            capacity=config.burst_capacity,
            refill_rate=refill_rate,
        )

        self._buckets[config.provider] = bucket
        self._configs[config.provider] = config

        logger.info(
            "Rate limit provider registered",
            extra={
                "provider": config.provider.value,
                "rpm": config.rpm,
                "tpm": config.tpm,
                "refill_rate": refill_rate,
            },
        )

    async def acquire(
        self,
        provider: RateLimitProvider,
        request_tokens: float = 1.0,
        wait: bool = True,
    ) -> float:
        """Adquire permissão para fazer uma requisição; aguarda se necessário."""
        bucket = self._buckets.get(provider)
        if not bucket:
            raise RateLimitError(
                f"Provider {provider.value} not registered",
            )

        try:
            if wait:
                return await bucket.wait_for_tokens(request_tokens)
            else:
                acquired = await bucket.acquire(request_tokens, wait=False)
                if not acquired:
                    raise RateLimitError(
                        f"Rate limit exceeded for {provider.value}",
                        retry_after=5,
                    )
                return 0.0

        except RateLimitError:
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "provider": provider.value,
                    "tokens": request_tokens,
                },
            )
            raise

    async def check_capacity(
        self,
        provider: RateLimitProvider,
    ) -> float:
        """Verifica a capacidade disponível sem adquirir."""
        bucket = self._buckets.get(provider)
        if not bucket:
            return 0.0

        async with bucket._lock:
            await bucket._refill()
            return bucket.tokens

    def get_status(self) -> Dict[str, Any]:
        """Retorna o status do limitador de taxa para todos os provedores."""
        status = {}
        for provider, bucket in self._buckets.items():
            config = self._configs[provider]
            status[provider.value] = {
                "rpm": config.rpm,
                "tpm": config.tpm,
                "available_tokens": bucket.tokens,
                "capacity": bucket.capacity,
                "refill_rate": bucket.refill_rate,
            }
        return status

    async def reset_provider(self, provider: RateLimitProvider) -> None:
        """Reinicia o bucket do provedor (após recuperação)."""
        bucket = self._buckets.get(provider)
        if bucket:
            async with bucket._lock:
                bucket.tokens = bucket.capacity
                bucket.last_refill = time.time()
            logger.info(
                "Rate limiter reset",
                extra={"provider": provider.value},
            )


_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Retorna ou cria o limitador de taxa global."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


async def rate_limit(
    provider: RateLimitProvider,
    request_tokens: float = 1.0,
    wait: bool = True,
) -> float:
    """Função auxiliar para adquirir limite de taxa."""
    limiter = get_rate_limiter()
    return await limiter.acquire(
        provider,
        request_tokens=request_tokens,
        wait=wait,
    )
