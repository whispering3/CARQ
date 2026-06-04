"""Pool de workers assíncronos com timeout e retry para execução de tarefas."""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Optional

from carq.core.exceptions import RetryableError, TaskTimeoutError
from carq.core.logging import get_logger
from carq.models.models import ProcessingTask, TaskType

logger = get_logger(__name__)


@dataclass
class WorkerConfig:
    """Configuração para um pool de workers."""

    worker_id: str
    max_workers: int = 4
    task_timeout: int = 300
    max_retries: int = 3
    task_types: list[TaskType] = None


class WorkerPool:
    """Gerencia um pool de workers assíncronos para execução de tarefas."""

    def __init__(
        self,
        config: Optional[WorkerConfig] = None,
        num_workers: int = 4,
        queue_size: int = 100,
    ):
        if config is None:
            config = WorkerConfig(
                worker_id=str(uuid.uuid4()),
                max_workers=num_workers,
            )
        self.config = config
        self.num_workers = num_workers
        self.queue_size = queue_size
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._running = False
        self._queue: asyncio.Queue = None
        self._active_tasks: dict[uuid.UUID, asyncio.Task] = {}
        self._task_handlers: dict[TaskType, Callable] = {}

    @property
    def is_running(self) -> bool:
        return self._running

    async def submit(self, coroutine) -> Any:
        """Submete uma corrotina para execução."""
        return await coroutine

    def register_handler(
        self,
        task_type: TaskType,
        handler: Callable[[ProcessingTask], Any],
    ) -> None:
        """Registra um handler para um tipo de tarefa."""
        self._task_handlers[task_type] = handler
        logger.info(
            "Handler registered",
            extra={
                "worker_id": self.config.worker_id,
                "task_type": task_type.value,
            },
        )

    async def execute_task(
        self,
        task: ProcessingTask,
        handler: Callable[[ProcessingTask], Any],
    ) -> Any:
        """Executa uma tarefa com timeout. Lança TaskTimeoutError se exceder o limite."""
        task_id = task.id
        self._active_tasks[task_id] = asyncio.current_task()

        try:
            result = await asyncio.wait_for(
                handler(task),
                timeout=self.config.task_timeout,
            )
            logger.info(
                "Task executed successfully",
                extra={
                    "task_id": str(task_id),
                    "task_type": task.task_type.value,
                    "worker_id": self.config.worker_id,
                },
            )
            return result

        except asyncio.TimeoutError:
            error = TaskTimeoutError(
                str(task_id),
                self.config.task_timeout,
            )
            logger.error(
                "Task timeout",
                extra={
                    "task_id": str(task_id),
                    "timeout": self.config.task_timeout,
                },
            )
            raise error

        except Exception as e:
            logger.error(
                "Task execution failed",
                extra={
                    "task_id": str(task_id),
                    "error": str(e),
                    "attempt": task.attempt_count,
                },
            )
            raise

        finally:
            self._active_tasks.pop(task_id, None)

    async def process_task(
        self,
        task: ProcessingTask,
    ) -> tuple[bool, Optional[str]]:
        """Processa uma tarefa com retry. Retorna (success, error_message)."""
        handler = self._task_handlers.get(task.task_type)
        if not handler:
            error_msg = f"No handler for task type: {task.task_type.value}"
            logger.error(
                error_msg,
                extra={
                    "task_id": str(task.id),
                    "task_type": task.task_type.value,
                },
            )
            return False, error_msg

        try:
            await self.execute_task(task, handler)
            return True, None

        except TaskTimeoutError as e:
            return False, str(e)

        except RetryableError as e:
            return False, str(e)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, error_msg

    async def start(self) -> None:
        """Inicia o pool de workers."""
        self._running = True
        logger.info(
            "Worker pool started",
            extra={
                "worker_id": self.config.worker_id,
                "max_workers": self.config.max_workers,
            },
        )

    async def stop(self, wait: bool = True) -> None:
        """Para o pool. Se wait=True, aguarda as tarefas ativas concluírem."""
        self._running = False

        if wait and self._active_tasks:
            logger.info(
                "Waiting for active tasks",
                extra={
                    "count": len(self._active_tasks),
                },
            )
            pending = list(self._active_tasks.values())
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        self.executor.shutdown(wait=wait)
        logger.info(
            "Worker pool stopped",
            extra={
                "worker_id": self.config.worker_id,
            },
        )

    @property
    def active_task_count(self) -> int:
        return len(self._active_tasks)

    def get_status(self) -> dict[str, Any]:
        return {
            "worker_id": self.config.worker_id,
            "running": self._running,
            "active_tasks": self.active_task_count,
            "max_workers": self.config.max_workers,
            "handlers": [t.value for t in self._task_handlers.keys()],
        }
