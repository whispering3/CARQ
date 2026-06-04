"""Padrão de circuit breaker para tolerância a falhas em cascata."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Coroutine, Optional, TypeVar

from carq.core.exceptions import CircuitBreakerOpenError
from carq.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """Estados do circuit breaker."""

    CLOSED = "CLOSED"  # Operação normal
    OPEN = "OPEN"  # Com falha, rejeita requisições
    HALF_OPEN = "HALF_OPEN"  # Testando recuperação


@dataclass
class CircuitBreakerConfig:
    """Configuração para o circuit breaker."""

    name: str
    failure_threshold: int = 5  # Falhas para abrir o circuito
    success_threshold: int = 2  # Sucessos para fechar o circuito
    timeout: int = 60  # Segundos antes de tentar recuperação
    half_open_max_calls: int = 3  # Máximo de chamadas no estado semi-aberto


class CircuitBreaker:
    """
    Circuit breaker para tolerância a falhas.

    Estados:
    - CLOSED: Operação normal, todas as requisições passam
    - OPEN: Serviço com falha, rejeita requisições imediatamente
    - HALF_OPEN: Testando recuperação, permite requisições limitadas

    Transições:
    - CLOSED → OPEN: Contagem de falhas excede o limite
    - OPEN → HALF_OPEN: Timeout expira
    - HALF_OPEN → CLOSED: Contagem de sucessos excede o limite
    - HALF_OPEN → OPEN: Falha no estado semi-aberto
    """

    def __init__(
        self,
        config: Optional[CircuitBreakerConfig] = None,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        success_threshold: int = 2,
        name: str = "default",
    ):
        """Inicializa o circuit breaker."""
        if config is None:
            config = CircuitBreakerConfig(
                name=name,
                failure_threshold=failure_threshold,
                success_threshold=success_threshold,
                timeout=recovery_timeout,
            )
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.last_open_time: Optional[float] = None
        self._lock = asyncio.Lock()
        self._half_open_calls = 0

    @property
    def failure_threshold(self) -> int:
        return self.config.failure_threshold

    @property
    def recovery_timeout(self) -> int:
        return self.config.timeout

    @property
    def success_threshold(self) -> int:
        return self.config.success_threshold

    async def _should_attempt_reset(self) -> bool:
        """Verifica se deve transicionar de OPEN para HALF_OPEN."""
        if self.state != CircuitState.OPEN:
            return False

        if not self.last_open_time:
            return False

        elapsed = time.time() - self.last_open_time
        return elapsed >= self.config.timeout

    async def _reset(self) -> None:
        """Reinicia os contadores e transiciona para HALF_OPEN."""
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitState.HALF_OPEN
        self._half_open_calls = 0

        logger.info(
            "Circuit breaker reset to half-open",
            extra={"breaker": self.config.name},
        )

    async def _open(self) -> None:
        """Abre o circuito após atingir o limite de falhas."""
        self.state = CircuitState.OPEN
        self.last_open_time = time.time()

        logger.warning(
            "Circuit breaker opened",
            extra={
                "breaker": self.config.name,
                "failure_count": self.failure_count,
                "threshold": self.config.failure_threshold,
            },
        )

    async def _close(self) -> None:
        """Fecha o circuito após sucesso no estado semi-aberto."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0

        logger.info(
            "Circuit breaker closed",
            extra={"breaker": self.config.name},
        )

    async def call(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> T:
        """Executa a função através do circuit breaker."""
        async with self._lock:
            if await self._should_attempt_reset():
                await self._reset()

            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit {self.config.name} is open",
                    retry_after=self.config.timeout,
                )

            if self.state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit {self.config.name} is open",
                        retry_after=10,
                    )
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                self.failure_count = 0
                self.success_count += 1

                if self.state == CircuitState.HALF_OPEN:
                    if self.success_count >= self.config.success_threshold:
                        await self._close()

            logger.debug(
                "Circuit breaker call succeeded",
                extra={
                    "breaker": self.config.name,
                    "state": self.state.value,
                    "success_count": self.success_count,
                },
            )

            return result

        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()

                logger.warning(
                    "Circuit breaker call failed",
                    extra={
                        "breaker": self.config.name,
                        "error": str(e),
                        "failure_count": self.failure_count,
                        "threshold": self.config.failure_threshold,
                    },
                )

                if (
                    self.state == CircuitState.CLOSED
                    and self.failure_count >= self.config.failure_threshold
                ):
                    await self._open()

                elif self.state == CircuitState.HALF_OPEN:
                    await self._open()

            raise

    def get_status(self) -> dict:
        """Retorna o status do circuit breaker."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.config.failure_threshold,
            "success_threshold": self.config.success_threshold,
            "timeout": self.config.timeout,
        }


class CircuitBreakerManager:
    """Gerencia múltiplos circuit breakers."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def register_breaker(self, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Registra um novo circuit breaker."""
        async with self._lock:
            if config.name in self._breakers:
                return self._breakers[config.name]

            breaker = CircuitBreaker(config)
            self._breakers[config.name] = breaker

            logger.info(
                "Circuit breaker registered",
                extra={
                    "name": config.name,
                    "failure_threshold": config.failure_threshold,
                    "timeout": config.timeout,
                },
            )

            return breaker

    async def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Retorna o circuit breaker pelo nome."""
        return self._breakers.get(name)

    async def call(
        self,
        breaker_name: str,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> T:
        """Executa a função com proteção do circuit breaker."""
        breaker = self._breakers.get(breaker_name)
        if not breaker:
            raise ValueError(f"Unknown circuit breaker: {breaker_name}")

        return await breaker.call(func, *args, **kwargs)

    def get_status(self) -> dict[str, dict]:
        """Retorna o status de todos os circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    async def reset_all(self) -> None:
        """Reinicia todos os circuit breakers para o estado CLOSED."""
        async with self._lock:
            for breaker in self._breakers.values():
                async with breaker._lock:
                    await breaker._close()

            logger.info("All circuit breakers reset")


# Gerenciador global de circuit breakers
_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Retorna ou cria o gerenciador global de circuit breakers."""
    global _manager
    if _manager is None:
        _manager = CircuitBreakerManager()
    return _manager


async def create_breaker(config: CircuitBreakerConfig) -> CircuitBreaker:
    """Auxiliar para criar um circuit breaker."""
    manager = get_circuit_breaker_manager()
    return await manager.register_breaker(config)
