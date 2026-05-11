"""Tratamento centralizado de erros com gerenciamento de Dead Letter Queue (DLQ)."""
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from carq.core.logging import get_logger
from carq.core.exceptions import ProcessingError
from carq.models.models import ProcessingTask, TaskStatus, Document, DocumentStatus

logger = get_logger(__name__)


class DLQEntry:
    """Representa uma entrada na Dead Letter Queue."""
    def __init__(
        self,
        task_id: str,
        document_id: str,
        task_type: str,
        error_details: dict,
        failed_at: datetime,
        attempt_count: int,
    ):
        self.task_id = task_id
        self.document_id = document_id
        self.task_type = task_type
        self.error_details = error_details
        self.failed_at = failed_at
        self.attempt_count = attempt_count

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "document_id": self.document_id,
            "task_type": self.task_type,
            "error_details": self.error_details,
            "failed_at": self.failed_at.isoformat(),
            "attempt_count": self.attempt_count,
        }


class ErrorHandler:
    """Tratamento centralizado de erros com gerenciamento de Dead Letter Queue."""

    def __init__(self, session: AsyncSession, max_attempts: int = 3):
        self.session = session
        self.max_attempts = max_attempts

    async def scan_and_move_to_dlq(self) -> List[DLQEntry]:
        """Verifica tarefas que excederam max_attempts e as marca como DLQ."""
        query = (
            select(ProcessingTask)
            .where(
                ProcessingTask.status == TaskStatus.FAILED,
                ProcessingTask.attempt_count >= self.max_attempts,
            )
        )
        result = await self.session.execute(query)
        failed_tasks = result.scalars().all()

        dlq_entries = []
        for task in failed_tasks:
            attrs = task.attributes or {}
            if not attrs.get("dlq"):
                attrs["dlq"] = True
                attrs["dlq_at"] = datetime.now(timezone.utc).isoformat()
                task.attributes = attrs
                self.session.add(task)

                entry = DLQEntry(
                    task_id=str(task.id),
                    document_id=str(task.document_id),
                    task_type=str(task.task_type),
                    error_details=attrs.get("error_details", {}),
                    failed_at=task.updated_at or datetime.now(timezone.utc),
                    attempt_count=task.attempt_count,
                )
                dlq_entries.append(entry)
                logger.warning(
                    "Task moved to DLQ",
                    extra={
                        "task_id": str(task.id),
                        "attempt_count": task.attempt_count,
                        "task_type": str(task.task_type),
                    },
                )

        if dlq_entries:
            await self.session.flush()

        return dlq_entries

    async def get_dlq_entries(self) -> List[DLQEntry]:
        """Retorna todas as entradas atuais do DLQ."""
        query = (
            select(ProcessingTask)
            .where(ProcessingTask.status == TaskStatus.FAILED)
            .order_by(ProcessingTask.updated_at.desc())
        )
        result = await self.session.execute(query)
        tasks = result.scalars().all()

        entries = []
        for task in tasks:
            attrs = task.attributes or {}
            if attrs.get("dlq"):
                entries.append(DLQEntry(
                    task_id=str(task.id),
                    document_id=str(task.document_id),
                    task_type=str(task.task_type),
                    error_details=attrs.get("error_details", {}),
                    failed_at=task.updated_at or datetime.now(timezone.utc),
                    attempt_count=task.attempt_count,
                ))
        return entries

    async def get_dlq_count(self) -> int:
        """Retorna a contagem de entradas do DLQ via um único COUNT (sem carregamento completo)."""
        result = await self.session.execute(
            select(func.count()).select_from(ProcessingTask).where(
                ProcessingTask.status == TaskStatus.FAILED,
            )
        )
        return result.scalar_one() or 0

    async def replay_task(self, task_id: str) -> bool:
        """Repete uma tarefa do DLQ redefinindo seu status para PENDING."""
        import uuid
        try:
            task = await self.session.get(ProcessingTask, uuid.UUID(task_id))
            if not task:
                logger.warning("DLQ replay: task not found", extra={"task_id": task_id})
                return False

            task.status = TaskStatus.PENDING
            task.attempt_count = 0
            attrs = task.attributes or {}
            attrs.pop("dlq", None)
            attrs.pop("dlq_at", None)
            task.attributes = attrs
            self.session.add(task)
            await self.session.flush()

            logger.info(
                "DLQ task requeued",
                extra={"task_id": task_id},
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to replay DLQ task",
                extra={"task_id": task_id, "error": str(e)},
            )
            return False

    async def replay_all(self) -> int:
        """Repete todas as entradas do DLQ. Retorna a contagem de tarefas repetidas."""
        entries = await self.get_dlq_entries()
        count = 0
        for entry in entries:
            if await self.replay_task(entry.task_id):
                count += 1
        return count

    async def get_error_report(self) -> Dict[str, Any]:
        """Gera um relatório abrangente de erros com contagens por status e entradas do DLQ."""
        status_query = select(ProcessingTask.status, func.count()).group_by(ProcessingTask.status)
        result = await self.session.execute(status_query)
        status_counts = {
            (s.value if hasattr(s, "value") else s): c
            for s, c in result.all()
        }

        dlq_entries = await self.get_dlq_entries()

        return {
            "total_tasks": sum(status_counts.values()),
            "by_status": status_counts,
            "dlq_count": len(dlq_entries),
            "dlq_entries": [e.to_dict() for e in dlq_entries[:10]],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    async def handle_exception(
        self,
        task: ProcessingTask,
        error: Exception,
        move_to_dlq_if_max_attempts: bool = True,
    ) -> bool:
        """Incrementa tentativas, marca como falha e opcionalmente move para o DLQ. Retorna True se foi ao DLQ."""
        task.attempt_count += 1
        task.status = TaskStatus.FAILED

        attrs = task.attributes or {}
        attrs["error_details"] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "attempt_number": task.attempt_count,
        }
        task.attributes = attrs
        self.session.add(task)

        moved_to_dlq = False
        if move_to_dlq_if_max_attempts and task.attempt_count >= self.max_attempts:
            attrs["dlq"] = True
            attrs["dlq_at"] = datetime.now(timezone.utc).isoformat()
            task.attributes = attrs
            moved_to_dlq = True
            logger.warning(
                "Task moved to DLQ after max attempts",
                extra={
                    "task_id": str(task.id),
                    "attempt_count": task.attempt_count,
                    "error": str(error),
                },
            )
        else:
            logger.warning(
                "Task failed, will retry",
                extra={
                    "task_id": str(task.id),
                    "attempt_count": task.attempt_count,
                    "max_attempts": self.max_attempts,
                    "error": str(error),
                },
            )

        await self.session.flush()
        return moved_to_dlq
