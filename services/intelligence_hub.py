from __future__ import annotations

from database.models import ShelfIssue, TaskRecord
from services.inventory_service import check_backstock
from services.planogram_service import resolve_location
from services.rota_service import assign_employee
from services.task_engine import create_task


RESTOCK_ACTIONS = {"RESTOCK"}
REARRANGE_ACTIONS = {"REARRANGE"}
AUDIT_ACTIONS = {"AUDIT"}


def build_tasks(issues: list[ShelfIssue], audit_location: str) -> list[TaskRecord]:
    generated_tasks: list[TaskRecord] = []

    for issue in issues:
        location, shelf_level = resolve_location(issue.product_name, audit_location, issue.shelf_level)
        action = issue.suggested_action
        assignee, assignment_reason = assign_employee(action, location)
        assignment_payload = {
            "assignee_id": assignee.employee_id if assignee else None,
            "assignee_name": assignee.name if assignee else None,
            "assignee_role": assignee.role if assignee else None,
            "assignment_reason": assignment_reason,
        }

        if action in RESTOCK_ACTIONS and issue.product_name:
            backstock = check_backstock(issue.product_name)
            quantity = max(1, issue.gap_units or 2)

            if backstock > 0:
                generated_tasks.append(
                    create_task(
                        task_type="RESTOCK",
                        priority=issue.severity,
                        product_name=issue.product_name,
                        quantity=min(quantity, backstock),
                        location=location,
                        shelf_level=shelf_level,
                        reason=issue.details,
                        **assignment_payload,
                    )
                )
            else:
                generated_tasks.append(
                    create_task(
                        task_type="AUDIT",
                        priority=issue.severity,
                        product_name=issue.product_name,
                        quantity=0,
                        location=location,
                        shelf_level=shelf_level,
                        reason=f"Backstock not found in inventory. {issue.details}",
                        **assignment_payload,
                    )
                )
            continue

        if action in REARRANGE_ACTIONS:
            generated_tasks.append(
                create_task(
                    task_type="REARRANGE",
                    priority=issue.severity,
                    product_name=issue.product_name,
                    quantity=0,
                    location=location,
                    shelf_level=shelf_level,
                    reason=issue.details,
                    **assignment_payload,
                )
            )
            continue

        if action in AUDIT_ACTIONS:
            generated_tasks.append(
                create_task(
                    task_type="AUDIT",
                    priority=issue.severity,
                    product_name=issue.product_name,
                    quantity=0,
                    location=location,
                    shelf_level=shelf_level,
                    reason=issue.details,
                    **assignment_payload,
                )
            )

    return generated_tasks
