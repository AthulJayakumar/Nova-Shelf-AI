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
    assignee_id: str | None = None,
    assignee_name: str | None = None,
    assignee_role: str | None = None,
    assignment_reason: str | None = None,
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
        assignee_id=assignee_id,
        assignee_name=assignee_name,
        assignee_role=assignee_role,
        assignment_reason=assignment_reason,
    )
    tasks.append(task)
    return task


def list_tasks() -> list[TaskRecord]:
    return tasks
