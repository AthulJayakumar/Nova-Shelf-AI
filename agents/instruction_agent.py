from __future__ import annotations

from database.models import TaskRecord, VisionShelfAudit


def generate_instruction(tasks: list[TaskRecord], audit: VisionShelfAudit) -> str:
    if not tasks:
        return f"{audit.location} looks shelf-ready. No urgent action is required."

    ordered_tasks = sorted(tasks, key=lambda task: (task.priority != "HIGH", task.task_type != "RESTOCK", task.id))
    snippets: list[str] = []

    for task in ordered_tasks[:4]:
        assignee_text = f" Assign to {task.assignee_name}." if task.assignee_name else ""

        if task.task_type == "RESTOCK":
            snippets.append(
                f"Restock {task.quantity} units of {task.product_name} on {task.shelf_level}.{assignee_text}"
            )
        elif task.task_type == "REARRANGE":
            if task.product_name:
                snippets.append(f"Front-face and realign {task.product_name} on {task.shelf_level}.{assignee_text}")
            else:
                snippets.append(f"Rearrange mixed products on {task.shelf_level}.{assignee_text}")
        else:
            snippets.append(f"Audit {task.shelf_level} for {task.product_name or 'unknown product'}.{assignee_text}")

    return f"Priority actions for {audit.location}: " + " ".join(snippets)
