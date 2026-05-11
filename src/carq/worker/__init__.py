"""Módulo de workers - Análise de PDFs, segmentação e execução de embeddings."""

from carq.worker.worker_pool import WorkerPool, WorkerConfig
from carq.worker.task_coordinator import TaskCoordinator, CoordinatorConfig

__all__ = [
    "WorkerPool",
    "WorkerConfig",
    "TaskCoordinator",
    "CoordinatorConfig",
]
