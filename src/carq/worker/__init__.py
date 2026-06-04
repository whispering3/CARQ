"""Módulo de workers - Análise de PDFs, segmentação e execução de embeddings."""

from carq.worker.task_coordinator import CoordinatorConfig, TaskCoordinator
from carq.worker.worker_pool import WorkerConfig, WorkerPool

__all__ = [
    "WorkerPool",
    "WorkerConfig",
    "TaskCoordinator",
    "CoordinatorConfig",
]
