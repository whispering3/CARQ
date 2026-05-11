"""Coordenador de tarefas: distribui trabalho para pools de workers e monitora seu ciclo de vida."""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Callable, Optional, Dict, List, TYPE_CHECKING
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from carq.core.logging import get_logger
from carq.models.models import ProcessingTask, TaskStatus, TaskType
from carq.queue.queue_manager import QueueManager
from carq.worker.worker_pool import WorkerPool, WorkerConfig

if TYPE_CHECKING:
    from carq.core.database import DatabaseManager

logger = get_logger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuração para o coordenador de tarefas."""

    dequeue_interval: int = 1  # Segundos entre tentativas de desenfileiramento
    stuck_task_timeout: int = 300  # Redefine tarefas presas após N segundos
    status_update_interval: int = 30  # Segundos entre atualizações de status
    enable_stuck_task_recovery: bool = True


class TaskCoordinator:
    """Coordena o processamento distribuído de tarefas entre pools de workers."""

    def __init__(
        self,
        session: Optional[AsyncSession] = None,
        config: Optional[CoordinatorConfig] = None,
        db_manager: Optional["DatabaseManager"] = None,
    ):
        """Inicializa o coordenador de tarefas.

        Args:
            session: AsyncSession compartilhada (legado; evitada em loops de background quando
                     db_manager é fornecido, pois compartilhar uma sessão entre corrotinas
                     concorrentes causa DetachedInstanceError / estado inconsistente).
            config: Configuração do coordenador.
            db_manager: Preferido — DatabaseManager usado para criar sessões isoladas por operação
                        dentro de cada loop de background, eliminando o compartilhamento de sessão.
        """
        self.session = session
        self.db_manager = db_manager
        self.config = config or CoordinatorConfig()
        self.queue_manager = QueueManager(session) if session else None

        self._running = False
        self._session_lock = asyncio.Lock()  # CRÍTICO: Protege acesso concorrente à sessão
        self._worker_pools: Dict[str, WorkerPool] = {}
        self._task_assignments: Dict[uuid.UUID, str] = {}  # task_id -> worker_id
        self._metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_retried": 0,
            "errors_total": 0,
        }
        self._background_tasks: List[asyncio.Task] = []

    def register_worker_pool(self, pool: WorkerPool) -> None:
        """Registra um pool de workers para distribuição de tarefas."""
        self._worker_pools[pool.config.worker_id] = pool
        logger.info(
            "Worker pool registered",
            extra={
                "worker_id": pool.config.worker_id,
                "max_workers": pool.config.max_workers,
            },
        )

    async def start(self) -> None:
        """Inicia o coordenador e as tarefas de background."""
        self._running = True

        for pool in self._worker_pools.values():
            await pool.start()
        dequeue_task = asyncio.create_task(self._dequeue_loop())
        stuck_task_task = asyncio.create_task(self._stuck_task_recovery_loop())
        status_task = asyncio.create_task(self._status_update_loop())

        self._background_tasks.extend([dequeue_task, stuck_task_task, status_task])

        logger.info(
            "Task coordinator started",
            extra={
                "worker_pools": len(self._worker_pools),
                "config": {
                    "dequeue_interval": self.config.dequeue_interval,
                    "stuck_task_timeout": self.config.stuck_task_timeout,
                },
            },
        )

    async def stop(self) -> None:
        """Para o coordenador e todos os pools de workers."""
        self._running = False

        for task in self._background_tasks:
            if not task.done():
                task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        for pool in self._worker_pools.values():
            await pool.stop(wait=True)

        logger.info(
            "Task coordinator stopped",
            extra={"metrics": self._metrics},
        )

    async def _dequeue_loop(self) -> None:
        """Loop de background para desenfileirar e distribuir tarefas continuamente."""
        while self._running:
            try:
                if self.db_manager:
                    # Caminho preferido: cada iteração usa sua própria sessão isolada.
                    async with self.db_manager.get_session() as session:
                        queue_manager = QueueManager(session)
                        await self._dequeue_iteration(queue_manager)
                elif self.queue_manager:
                    # Caminho legado: sessão compartilhada (apenas setups de worker único).
                    await self._dequeue_iteration(self.queue_manager)

                await asyncio.sleep(self.config.dequeue_interval)

            except Exception as e:
                logger.error(
                    "Error in dequeue loop",
                    extra={"error": str(e)},
                )
                await asyncio.sleep(self.config.dequeue_interval)

    async def _dequeue_iteration(self, queue_manager: QueueManager) -> None:
        """Passagem única de desenfileiramento — chamada com um queue_manager já contextualizado."""
        for worker_pool in self._worker_pools.values():
            if not self._running:
                break

            tasks = await queue_manager.dequeue_task(
                worker_id=worker_pool.config.worker_id,
                task_types=worker_pool.config.task_types,
                batch_size=worker_pool.config.max_workers,
            )

            if tasks:
                logger.debug(
                    "Processing dequeued tasks",
                    extra={
                        "worker_id": worker_pool.config.worker_id,
                        "count": len(tasks),
                    },
                )
                await self._process_tasks(tasks, worker_pool, queue_manager)

    async def _process_tasks(
        self,
        tasks: List[ProcessingTask],
        pool: WorkerPool,
        queue_manager: Optional[QueueManager] = None,
    ) -> None:
        """
        Processa um lote de tarefas usando o pool de workers fornecido.

        Args:
            tasks: Lista de tarefas a processar
            pool: Pool de workers a usar para processamento
            queue_manager: Gerenciador de fila com escopo para a sessão da operação atual.
                           Usa self.queue_manager como fallback para chamadores legados.
        """
        qm = queue_manager or self.queue_manager
        for task in tasks:
            try:
                success, error = await pool.process_task(task)

                if success:
                    await qm.mark_task_done(task.id)
                    self._metrics["tasks_completed"] += 1

                else:
                    await qm.mark_task_failed(
                        task.id,
                        error or "Unknown error",
                    )
                    if task.attempt_count < task.max_attempts:
                        self._metrics["tasks_retried"] += 1
                    else:
                        self._metrics["tasks_failed"] += 1

                    self._metrics["errors_total"] += 1

                # Faz commit apenas ao usar o caminho legado de sessão compartilhada.
                # Quando db_manager é usado, o gerenciador de contexto da sessão faz commit automaticamente.
                if not self.db_manager and self.session:
                    async with self._session_lock:
                        await self.session.commit()

            except Exception as e:
                logger.error(
                    "Unexpected error processing task",
                    extra={
                        "task_id": str(task.id),
                        "error": str(e),
                    },
                )
                self._metrics["errors_total"] += 1

    async def _stuck_task_recovery_loop(self) -> None:
        """Loop de background para recuperar tarefas presas."""
        if not self.config.enable_stuck_task_recovery:
            return

        while self._running:
            try:
                if self.db_manager:
                    async with self.db_manager.get_session() as session:
                        qm = QueueManager(session)
                        count = await qm.reset_stuck_tasks(
                            timeout_seconds=self.config.stuck_task_timeout,
                        )
                elif self.queue_manager:
                    count = await self.queue_manager.reset_stuck_tasks(
                        timeout_seconds=self.config.stuck_task_timeout,
                    )
                    if count > 0:
                        async with self._session_lock:
                            await self.session.commit()
                else:
                    count = 0

                if count > 0:
                    logger.info(
                        "Stuck tasks recovered",
                        extra={"count": count},
                    )

                await asyncio.sleep(60)

            except Exception as e:
                logger.error(
                    "Error in stuck task recovery",
                    extra={"error": str(e)},
                )
                await asyncio.sleep(60)

    async def _status_update_loop(self) -> None:
        """Loop de background para registrar periodicamente o status do coordenador."""
        while self._running:
            try:
                if self.db_manager:
                    async with self.db_manager.get_session() as session:
                        qm = QueueManager(session)
                        queue_status = await qm.get_queue_status()
                elif self.queue_manager:
                    queue_status = await self.queue_manager.get_queue_status()
                else:
                    queue_status = {}

                logger.info(
                    "Coordinator status",
                    extra={
                        "queue": queue_status,
                        "metrics": self._metrics,
                        "worker_pools": {
                            pool_id: pool.get_status()
                            for pool_id, pool in self._worker_pools.items()
                        },
                    },
                )

                await asyncio.sleep(self.config.status_update_interval)

            except Exception as e:
                logger.error(
                    "Error in status update loop",
                    extra={"error": str(e)},
                )
                await asyncio.sleep(self.config.status_update_interval)

    async def assign_task(self, worker_id: str, task_data: dict) -> dict:
        task_id = str(uuid.uuid4())
        self._task_assignments[uuid.UUID(task_id)] = worker_id
        return {"task_id": task_id, "worker_id": worker_id, "status": "assigned"}

    async def record_completion(self, task_id: str, success: bool = True) -> None:
        if success:
            self._metrics["tasks_completed"] += 1
        else:
            self._metrics["tasks_failed"] += 1

    async def get_task_status(self, task_id: str) -> dict:
        task_uuid = uuid.UUID(task_id)
        worker_id = self._task_assignments.get(task_uuid)
        return {
            "task_id": task_id,
            "worker_id": worker_id,
            "status": "assigned" if worker_id else "unknown",
        }

    def get_metrics(self) -> dict:
        return {
            **self._metrics,
            "active_workers": sum(
                pool.active_task_count for pool in self._worker_pools.values()
            ),
            "total_worker_pools": len(self._worker_pools),
        }

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "worker_pools": {
                pool_id: pool.get_status()
                for pool_id, pool in self._worker_pools.items()
            },
            "metrics": self.get_metrics(),
            "background_tasks": len(self._background_tasks),
        }
