"""Sistema de verificação de saúde com probes de prontidão e vivacidade."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Status de verificação de saúde."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Resultado individual de verificação de saúde."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0


@dataclass
class HealthResponse:
    """Resposta geral de saúde."""

    status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    ready: bool = False
    alive: bool = True
    message: str = ""

    @property
    def overall_status(self) -> HealthStatus:
        """Determina o status geral a partir das verificações."""
        if not self.checks:
            return self.status

        statuses = [check.status for check in self.checks]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED


class HealthChecker:
    """Realiza verificações de saúde nas dependências do sistema (PostgreSQL, Redis)."""

    def __init__(
        self,
        db_session: Optional[AsyncSession] = None,
        redis_url: Optional[str] = None,
        timeout_seconds: float = 5.0,
    ):
        self.db_session = db_session
        self.redis_url = redis_url
        self.timeout_seconds = timeout_seconds
        self.logger = logging.getLogger(__name__)

    async def check_postgres(self) -> HealthCheck:
        """Verifica a saúde do banco de dados PostgreSQL."""
        start = time.monotonic()

        try:
            if not self.db_session:
                return HealthCheck(
                    name="postgres",
                    status=HealthStatus.UNHEALTHY,
                    message="Database session not configured",
                )

            await asyncio.wait_for(
                self.db_session.execute(text("SELECT 1")),
                timeout=self.timeout_seconds,
            )

            latency = (time.monotonic() - start) * 1000

            return HealthCheck(
                name="postgres",
                status=HealthStatus.HEALTHY,
                message="Connected",
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return HealthCheck(
                name="postgres",
                status=HealthStatus.UNHEALTHY,
                message="Connection timeout",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            self.logger.warning(f"PostgreSQL health check failed: {e}")
            return HealthCheck(
                name="postgres",
                status=HealthStatus.UNHEALTHY,
                message=f"Error: {str(e)[:50]}",
                latency_ms=(time.monotonic() - start) * 1000,
            )

    async def check_redis(self) -> HealthCheck:
        """Verifica a saúde do cache Redis."""
        start = time.monotonic()

        try:
            if not self.redis_url:
                return HealthCheck(
                    name="redis",
                    status=HealthStatus.DEGRADED,
                    message="Redis not configured",
                )

            client = await redis.from_url(self.redis_url)
            await asyncio.wait_for(
                client.ping(),
                timeout=self.timeout_seconds,
            )
            await client.close()

            latency = (time.monotonic() - start) * 1000

            return HealthCheck(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Connected",
                latency_ms=latency,
            )

        except asyncio.TimeoutError:
            return HealthCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message="Connection timeout",
                latency_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as e:
            self.logger.warning(f"Redis health check failed: {e}")
            return HealthCheck(
                name="redis",
                status=HealthStatus.DEGRADED,
                message=f"Error: {str(e)[:50]}",
                latency_ms=(time.monotonic() - start) * 1000,
            )

    async def check_readiness(self) -> HealthResponse:
        """Prontidão = todas as dependências saudáveis."""
        start = time.monotonic()

        checks = [
            await self.check_postgres(),
            await self.check_redis(),
        ]

        latency = (time.monotonic() - start) * 1000

        ready = all(c.status == HealthStatus.HEALTHY for c in checks)

        if ready:
            status = HealthStatus.HEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in checks):
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.DEGRADED

        return HealthResponse(
            status=status,
            checks=checks,
            ready=ready,
            alive=True,
            message=f"Readiness check took {latency:.1f}ms",
        )

    async def check_liveness(self) -> HealthResponse:
        """Vivacidade = processo em execução (sempre retorna alive=True)."""
        checks = [
            HealthCheck(
                name="process",
                status=HealthStatus.HEALTHY,
                message="Process running",
            ),
        ]

        return HealthResponse(
            status=HealthStatus.HEALTHY,
            checks=checks,
            ready=True,
            alive=True,
            message="Service is alive",
        )

    async def check_full(self) -> HealthResponse:
        """Verificação completa de saúde (prontidão + vivacidade)."""
        readiness = await self.check_readiness()
        liveness = await self.check_liveness()

        all_checks = readiness.checks + liveness.checks

        status = HealthStatus.HEALTHY
        if any(c.status == HealthStatus.UNHEALTHY for c in all_checks):
            status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in all_checks):
            status = HealthStatus.DEGRADED

        return HealthResponse(
            status=status,
            checks=all_checks,
            ready=readiness.ready,
            alive=liveness.alive,
            message="Full health check complete",
        )


# Instância global do verificador de saúde
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Obtém ou cria o verificador de saúde global (singleton)."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker
