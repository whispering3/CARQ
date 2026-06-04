"""Estratégias de backoff para lógica de retry, com jitter para evitar thundering herd."""

import asyncio
import random
from dataclasses import dataclass
from typing import Optional

from carq.core.exceptions import TaskTimeoutError
from carq.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BackoffConfig:
    """Configuração para estratégia de backoff."""

    initial_delay: float = 1.0  # Segundos
    max_delay: float = 300.0  # 5 minutos
    exponential_base: float = 2.0
    jitter_factor: float = 0.1  # 10% de jitter


class BackoffStrategy:
    """Estratégia de backoff configurável (exponencial por padrão)."""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 300.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        jitter_factor: float = 0.1,
    ):
        """Inicializa a estratégia de backoff com parâmetros simples."""
        self.config = BackoffConfig(
            initial_delay=initial_delay,
            max_delay=max_delay,
            exponential_base=exponential_base,
            jitter_factor=jitter_factor if jitter else 0.0,
        )
        self._logger = get_logger(__name__)

    def get_delay(self, attempt: int) -> float:
        """Calcula o atraso de backoff exponencial."""
        base_delay = self.config.initial_delay * (self.config.exponential_base ** attempt)
        delay = min(base_delay, self.config.max_delay)
        jitter = random.uniform(0, delay * self.config.jitter_factor)
        return delay + jitter

    def calculate_delay(self, attempt: int) -> float:
        """Alias para get_delay."""
        return self.get_delay(attempt)

    @property
    def initial_delay(self) -> float:
        return self.config.initial_delay

    @property
    def max_delay(self) -> float:
        return self.config.max_delay

    async def wait(self, attempt: int) -> float:
        """Aguarda o atraso de backoff."""
        delay = self.get_delay(attempt)
        await asyncio.sleep(delay)
        return delay


class ExponentialBackoff(BackoffStrategy):
    """
    Backoff exponencial com jitter opcional.

    Atraso = min(initial_delay * (base ^ tentativa), max_delay)
    Jitter = atraso * random(0, jitter_factor)

    Benefícios:
    - Aumenta o atraso exponencialmente para reduzir a carga
    - Jitter evita thundering herd (todas as tentativas ao mesmo tempo)
    - Atrasos mínimos/máximos configuráveis
    """

    def __init__(self, config: BackoffConfig):
        self.config = config
        self._logger = get_logger(__name__)

    def get_delay(self, attempt: int) -> float:
        """Calcula o atraso de backoff exponencial."""
        base_delay = self.config.initial_delay * (
            self.config.exponential_base ** attempt
        )

        delay = min(base_delay, self.config.max_delay)

        jitter = random.uniform(0, delay * self.config.jitter_factor)

        return delay + jitter

    async def wait(self, attempt: int) -> float:
        """Aguarda o atraso de backoff exponencial."""
        delay = self.get_delay(attempt)

        self._logger.debug(
            "Exponential backoff",
            extra={
                "attempt": attempt,
                "delay": delay,
                "initial": self.config.initial_delay,
                "base": self.config.exponential_base,
            },
        )

        await asyncio.sleep(delay)
        return delay


class LinearBackoff(BackoffStrategy):
    """
    Estratégia de backoff linear.

    Atraso = min(initial_delay * tentativa, max_delay)
    """

    def __init__(self, config: BackoffConfig):
        self.config = config
        self._logger = get_logger(__name__)

    def get_delay(self, attempt: int) -> float:
        """Calcula o atraso de backoff linear."""
        delay = min(
            self.config.initial_delay * (attempt + 1),
            self.config.max_delay,
        )

        jitter = random.uniform(0, delay * self.config.jitter_factor)
        return delay + jitter

    async def wait(self, attempt: int) -> float:
        """Aguarda o atraso de backoff linear."""
        delay = self.get_delay(attempt)

        self._logger.debug(
            "Linear backoff",
            extra={
                "attempt": attempt,
                "delay": delay,
            },
        )

        await asyncio.sleep(delay)
        return delay


class FixedBackoff(BackoffStrategy):
    """Backoff com atraso fixo (sem aumento exponencial)."""

    def __init__(self, config: BackoffConfig):
        self.config = config
        self._logger = get_logger(__name__)

    def get_delay(self, attempt: int) -> float:
        """Retorna o atraso fixo."""
        return self.config.initial_delay

    async def wait(self, attempt: int) -> float:
        """Aguarda o atraso fixo."""
        delay = self.get_delay(attempt)
        await asyncio.sleep(delay)
        return delay


class RetryPolicy:
    """
    Política de retry combinando backoff e máximo de tentativas.

    Gerencia retentativas com backoff exponencial até atingir o máximo de tentativas.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff: Optional[BackoffStrategy] = None,
    ):
        self.max_attempts = max_attempts
        self.backoff = backoff or ExponentialBackoff(BackoffConfig())
        self._logger = get_logger(__name__)

    async def execute_with_retry(
        self,
        func,
        *args,
        **kwargs,
    ):
        """Executa a função com lógica de retry até max_attempts."""
        last_exception = None

        for attempt in range(self.max_attempts):
            try:
                self._logger.debug(
                    "Executing with retry",
                    extra={
                        "attempt": attempt + 1,
                        "max_attempts": self.max_attempts,
                    },
                )

                result = await func(*args, **kwargs)
                return result

            except TaskTimeoutError as e:
                last_exception = e

                if attempt < self.max_attempts - 1:
                    delay = await self.backoff.wait(attempt)
                    self._logger.warning(
                        "Task timeout, retrying",
                        extra={
                            "attempt": attempt + 1,
                            "delay": delay,
                            "error": str(e),
                        },
                    )
                else:
                    self._logger.error(
                        "Task timeout after max retries",
                        extra={
                            "attempts": self.max_attempts,
                            "error": str(e),
                        },
                    )

            except Exception as e:
                if self._is_retryable(e) and attempt < self.max_attempts - 1:
                    last_exception = e
                    delay = await self.backoff.wait(attempt)

                    self._logger.warning(
                        "Retryable error, backing off",
                        extra={
                            "attempt": attempt + 1,
                            "delay": delay,
                            "error": str(e),
                        },
                    )
                else:
                    self._logger.error(
                        "Non-retryable error or max attempts reached",
                        extra={
                            "attempt": attempt + 1,
                            "attempts_left": self.max_attempts - attempt - 1,
                            "error": str(e),
                        },
                    )
                    raise

        # Todas as tentativas falharam
        if last_exception:
            raise last_exception

    @staticmethod
    def _is_retryable(exception: Exception) -> bool:
        """Verifica se a exceção é passível de retry."""
        retryable_types = (
            TaskTimeoutError,
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        )
        return isinstance(exception, retryable_types)

    def get_config(self) -> dict:
        """Retorna a configuração da política de retry."""
        return {
            "max_attempts": self.max_attempts,
            "backoff_strategy": self.backoff.__class__.__name__,
        }


async def retry_with_backoff(
    func,
    max_attempts: int = 3,
    backoff_config: Optional[BackoffConfig] = None,
    *args,
    **kwargs,
):
    """
    Função auxiliar para retry com backoff exponencial.

    Uso:
        result = await retry_with_backoff(
            my_async_func,
            max_attempts=3,
            arg1="value",
            kwarg="value",
        )
    """
    config = backoff_config or BackoffConfig()
    backoff = ExponentialBackoff(config)
    policy = RetryPolicy(max_attempts=max_attempts, backoff=backoff)

    return await policy.execute_with_retry(func, *args, **kwargs)
