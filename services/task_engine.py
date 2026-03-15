from __future__ import annotations

from itertools import count

from database.models import TaskRecord


_task_ids = count(1)
tasks: list[TaskRecord] = []


def create_task(
    *,
    task_type: str,
    priority: str,
    product_name: str | None,
    quantity: int,
    location: str,
    shelf_level: str,
    reason: str,
) -> TaskRecord:
    task = TaskRecord(
        id=next(_task_ids),
        task_type=task_type,
        priority=priority,
        product_name=product_name,
        quantity=quantity,
        location=location,
        shelf_level=shelf_level,
        reason=reason,
    )
    tasks.append(task)
    return task


def list_tasks() -> list[TaskRecord]:
    return tasks
