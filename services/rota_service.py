from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from itertools import count
from pathlib import Path

from database.models import EmployeeRecord


ROTA_PATH = Path(__file__).resolve().parent.parent / "data" / "rota.json"
DAY_CODES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
_assignment_counter = count(0)


@lru_cache
def get_rota() -> list[EmployeeRecord]:
    raw = json.loads(ROTA_PATH.read_text(encoding="utf-8-sig"))
    employees = raw.get("employees", [])
    return [EmployeeRecord(**employee) for employee in employees]


def list_rota() -> list[EmployeeRecord]:
    return get_rota()


def get_active_employees(at_time: datetime | None = None) -> list[EmployeeRecord]:
    moment = at_time or datetime.now().astimezone()
    day_code = DAY_CODES[moment.weekday()]
    current_minutes = (moment.hour * 60) + moment.minute

    active: list[EmployeeRecord] = []
    for employee in get_rota():
        for shift in employee.shifts:
            if shift.day != day_code:
                continue
            start_minutes = _to_minutes(shift.start)
            end_minutes = _to_minutes(shift.end)
            if start_minutes <= current_minutes < end_minutes:
                active.append(employee)
                break

    return active


def assign_employee(task_type: str, location: str, at_time: datetime | None = None) -> tuple[EmployeeRecord | None, str]:
    active_employees = get_active_employees(at_time=at_time)
    if not active_employees:
        return None, "No employee is currently on shift in the rota."

    skill_matched = [employee for employee in active_employees if "ALL" in employee.skills or task_type in employee.skills]
    zone_matched = [employee for employee in skill_matched if not employee.zones or location in employee.zones]
    candidates = zone_matched or skill_matched or active_employees

    index = next(_assignment_counter) % len(candidates)
    employee = candidates[index]

    if zone_matched:
        return employee, f"Assigned by active shift, skill match, and zone match for {location}."
    if skill_matched:
        return employee, "Assigned by active shift and task skill match."
    return employee, "Assigned by active shift availability."



def _to_minutes(value: str) -> int:
    hours, minutes = value.split(":", 1)
    return (int(hours) * 60) + int(minutes)
