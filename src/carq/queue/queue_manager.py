"""Gerenciamento de filas com SELECT ... FOR UPDATE SKIP LOCKED para desenfileiramento sem colisões."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import and_, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from carq.core.exceptions import (
    TaskAlreadyProcessingError,
    TaskNotFoundError,
    QueueError,
)
from carq.core.logging import get_logger
from carq.models.models import (
    ProcessingTask,
    TaskStatus,
    TaskType,
    Document,
)

logger = get_logger(__name__)


class QueueManager:
    """Gerencia a fila assíncrona de tarefas com PostgreSQL."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def enqueue_task(
        self,
        document_id: uuid.UUID,
        task_type: TaskType,
        priority: int = 0,
        metadata: Optional[dict] = None,
    ) -> ProcessingTask:
        """Enfileira uma nova tarefa de processamento."""
        try:
            task = ProcessingTask(
                document_id=document_id,
                task_type=task_type,
                priority=priority,
                attributes=metadata or {},
                status=TaskStatus.PENDING,
            )
            self.session.add(task)
            await self.session.flush()

            logger.info(
                "Task enqueued",
                extra={
                    "task_id": str(task.id),
                    "document_id": str(document_id),
                    "task_type": task_type.value,
                    "priority": priority,
                },
            )

            return task

        except Exception as e:
            logger.error(
                "Failed to enqueue task",
                extra={
                    "error": str(e),
                    "document_id": str(document_id),
                },
            )
            raise QueueError(f"Failed to enqueue task: {str(e)}")

    async def dequeue_task(
        self,
        worker_id: str,
        task_types: Optional[list[TaskType]] = None,
        batch_size: int = 1,
    ) -> list[ProcessingTask]:
        """
        Desenfileira tarefas usando SELECT ... FOR UPDATE SKIP LOCKED.

        Garante que apenas um worker pode reivindicar uma tarefa e que tarefas
        travadas são ignoradas (SKIP LOCKED), sem condições de corrida.
        """
        try:
            query = (
                select(ProcessingTask)
                .where(ProcessingTask.status == TaskStatus.PENDING)
                .where(ProcessingTask.attempt_count < ProcessingTask.max_attempts)
                .order_by(ProcessingTask.priority.desc(), ProcessingTask.created_at.asc())
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )

            if task_types:
                task_type_values = [t.value for t in task_types]
                query = query.where(ProcessingTask.task_type.in_(task_type_values))

            result = await self.session.execute(query)
            tasks = result.scalars().all()

            for task in tasks:
                task.status = TaskStatus.PROCESSING
                task.worker_id = worker_id
                task.attempt_count += 1

            await self.session.flush()

            if tasks:
                logger.info(
                    "Tasks dequeued",
                    extra={
                        "worker_id": worker_id,
                        "count": len(tasks),
                        "batch_size": batch_size,
                    },
                )

            return tasks

        except Exception as e:
            logger.error(
                "Failed to dequeue tasks",
                extra={
                    "error": str(e),
                    "worker_id": worker_id,
                },
            )
            raise QueueError(f"Failed to dequeue tasks: {str(e)}")

    async def mark_task_done(
        self,
        task_id: uuid.UUID,
        metadata: Optional[dict] = None,
    ) -> ProcessingTask:
        """Marca a tarefa como concluída com sucesso."""
        try:
            task = await self.session.get(ProcessingTask, task_id)
            if not task:
                raise TaskNotFoundError(str(task_id))

            task.status = TaskStatus.DONE
            task.completed_at = datetime.now(timezone.utc)
            if metadata:
                task.attributes.update(metadata)

            await self.session.flush()

            logger.info(
                "Task marked as done",
                extra={
                    "task_id": str(task_id),
                    "document_id": str(task.document_id),
                },
            )

            return task

        except Exception as e:
            logger.error(
                "Failed to mark task as done",
                extra={
                    "error": str(e),
                    "task_id": str(task_id),
                },
            )
            raise

    async def mark_task_failed(
        self,
        task_id: uuid.UUID,
        error_message: str,
        error_details: Optional[dict] = None,
    ) -> ProcessingTask:
        """Marca a tarefa como falha, ou como RETRYING se ainda tem tentativas."""
        try:
            task = await self.session.get(ProcessingTask, task_id)
            if not task:
                raise TaskNotFoundError(str(task_id))

            task.error_message = error_message
            if error_details:
                task.attributes["error_details"] = error_details

            if task.is_retryable:
                task.status = TaskStatus.RETRYING
            else:
                task.status = TaskStatus.FAILED

            await self.session.flush()

            logger.warning(
                "Task marked as failed",
                extra={
                    "task_id": str(task_id),
                    "status": task.status.value,
                    "attempts": task.attempt_count,
                    "max_attempts": task.max_attempts,
                    "error": error_message,
                },
            )

            return task

        except Exception as e:
            logger.error(
                "Failed to mark task as failed",
                extra={
                    "error": str(e),
                    "task_id": str(task_id),
                },
            )
            raise

    async def reset_stuck_tasks(
        self,
        timeout_seconds: int = 300,
    ) -> int:
        """Reinicia tarefas presas em PROCESSING há mais que timeout_seconds (recuperação de crash)."""
        try:
            from datetime import datetime, timedelta, timezone

            cutoff_time = datetime.now(timezone.utc) - timedelta(seconds=timeout_seconds)

            query = update(ProcessingTask).where(
                and_(
                    ProcessingTask.status == TaskStatus.PROCESSING,
                    ProcessingTask.updated_at < cutoff_time,
                )
            ).values(
                status=TaskStatus.PENDING,
                worker_id=None,
            )

            result = await self.session.execute(query)
            await self.session.flush()

            count = result.rowcount
            logger.warning(
                "Reset stuck tasks",
                extra={
                    "count": count,
                    "timeout_seconds": timeout_seconds,
                },
            )

            return count

        except Exception as e:
            logger.error(
                "Failed to reset stuck tasks",
                extra={
                    "error": str(e),
                },
            )
            raise QueueError(f"Failed to reset stuck tasks: {str(e)}")

    async def mark_tasks_done(self, task_ids: list[uuid.UUID]) -> int:
        """Marca tarefas em lote como concluídas.
        
        CORREÇÃO CRÍTICA: Substitui N+1 atualizações individuais por atualização em lote.
        """
        if not task_ids:
            return 0
        
        try:
            from datetime import datetime, timezone
            
            query = (
                update(ProcessingTask)
                .where(ProcessingTask.id.in_(task_ids))
                .values(
                    status=TaskStatus.DONE,
                    completed_at=datetime.now(timezone.utc),
                )
            )
            
            result = await self.session.execute(query)
            count = result.rowcount
            
            logger.info(
                "Batch marked tasks as done",
                extra={
                    "count": count,
                    "requested": len(task_ids),
                },
            )
            
            return count
        
        except Exception as e:
            logger.error(
                "Failed to batch mark tasks as done",
                extra={
                    "error": str(e),
                    "count": len(task_ids),
                },
            )
            raise QueueError(f"Failed to mark tasks as done: {str(e)}")

    async def mark_tasks_failed(
        self,
        task_ids: list[uuid.UUID],
        error_message: str = "Batch processing failed",
    ) -> int:
        """Marca tarefas em lote como falhas.
        
        CORREÇÃO CRÍTICA: Substitui N+1 atualizações individuais por atualização em lote.
        """
        if not task_ids:
            return 0
        
        try:
            query = (
                update(ProcessingTask)
                .where(ProcessingTask.id.in_(task_ids))
                .values(
                    status=TaskStatus.FAILED,
                    error_message=error_message,
                )
            )
            
            result = await self.session.execute(query)
            count = result.rowcount
            
            logger.warning(
                "Batch marked tasks as failed",
                extra={
                    "count": count,
                    "requested": len(task_ids),
                    "error": error_message,
                },
            )
            
            return count
        
        except Exception as e:
            logger.error(
                "Failed to batch mark tasks as failed",
                extra={
                    "error": str(e),
                    "count": len(task_ids),
                },
            )
            raise QueueError(f"Failed to mark tasks as failed: {str(e)}")

    async def get_queue_status(self) -> dict:
        """Retorna estatísticas atuais da fila."""
        try:
            query = select(ProcessingTask.status, func.count()).group_by(
                ProcessingTask.status
            )
            result = await self.session.execute(query)
            status_counts = dict(result.all())

            doc_query = select(
                Document.status,
                func.count(),
            ).group_by(Document.status)
            doc_result = await self.session.execute(doc_query)
            doc_counts = dict(doc_result.all())

            return {
                "tasks": {(s.value if hasattr(s, "value") else s): count for s, count in status_counts.items()},
                "documents": {(s.value if hasattr(s, "value") else s): count for s, count in doc_counts.items()},
                "total_tasks": sum(status_counts.values()),
                "total_documents": sum(doc_counts.values()),
            }

        except Exception as e:
            logger.error(
                "Failed to get queue status",
                extra={
                    "error": str(e),
                },
            )
            return {}

    async def get_task(self, task_id: uuid.UUID) -> Optional[ProcessingTask]:
        """Retorna a tarefa pelo ID."""
        return await self.session.get(ProcessingTask, task_id)

    async def get_document_tasks(
        self,
        document_id: uuid.UUID,
        status: Optional[TaskStatus] = None,
    ) -> list[ProcessingTask]:
        """Retorna todas as tarefas de um documento."""
        query = select(ProcessingTask).where(
            ProcessingTask.document_id == document_id
        )
        if status:
            query = query.where(ProcessingTask.status == status)

        result = await self.session.execute(query)
        return result.scalars().all()
