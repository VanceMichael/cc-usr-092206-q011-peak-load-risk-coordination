"""需求响应协调。

- 按预警缺口向企业分配削峰指令，容量台账按 (地区, 时窗) 记账。
- 企业退出响应或调度撤回指令时，已分配容量立即恢复（台账当日当时即释放，
  并写入审计事件），可重新用于后续分配。
- 企业视图仅返回本企业的行动要求，不暴露其他企业的指令与容量。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from .models import (
    Alert,
    Enterprise,
    Instruction,
    InstructionStatus,
    TERMINAL_STATUSES,
    Window,
)


class ResponseCoordinator:
    def __init__(self, enterprises: tuple[Enterprise, ...]) -> None:
        self._enterprises = {e.id: e for e in enterprises}
        self._instructions: dict[str, Instruction] = {}
        self._audit: list[tuple[datetime, str, dict]] = []
        self._seq = 0

    # ---- 计划与签发 ----

    def plan(self, alert: Alert, now: datetime) -> tuple[Instruction, ...]:
        """按预警缺口向该地区企业分配削峰量（按可削减能力从大到小）。

        缺口先扣除该地区该时窗已承诺的容量，避免重复规划时超额分配。
        """
        gap = max(0.0, -alert.reserve_margin_mw)
        remaining = gap - self.committed_mw(alert.region, alert.window)
        if remaining <= 0.0:
            return ()
        candidates = sorted(
            (e for e in self._enterprises.values() if e.region == alert.region),
            key=lambda e: e.max_reduction_mw,
            reverse=True,
        )
        issued: list[Instruction] = []
        for enterprise in candidates:
            if remaining <= 0.0:
                break
            already = self._committed_by(enterprise.id, alert.region, alert.window)
            headroom = enterprise.max_reduction_mw - already
            if headroom <= 0.0:
                continue
            amount = min(headroom, remaining)
            issued.append(self._issue(enterprise, alert, amount, now))
            remaining -= amount
        return tuple(issued)

    def _issue(
        self, enterprise: Enterprise, alert: Alert, amount: float, now: datetime
    ) -> Instruction:
        self._seq += 1
        instruction = Instruction(
            id=f"IN-{self._seq:04d}",
            enterprise_id=enterprise.id,
            region=alert.region,
            window=alert.window,
            requested_reduction_mw=amount,
            status=InstructionStatus.ISSUED,
            issued_at=now,
            alert_id=alert.id,
            history=((now, "issued"),),
        )
        self._instructions[instruction.id] = instruction
        self._audit.append(
            (
                now,
                "issued",
                {
                    "instruction_id": instruction.id,
                    "enterprise_id": enterprise.id,
                    "window": alert.window.key,
                    "requested_mw": amount,
                    "alert_id": alert.id,
                },
            )
        )
        return instruction

    # ---- 状态迁移 ----

    def acknowledge(self, instruction_id: str, now: datetime) -> Instruction:
        instruction = self._require(instruction_id)
        self._ensure_active(instruction)
        return self._transition(instruction, InstructionStatus.ACKNOWLEDGED, now, "acknowledged")

    def enterprise_exit(self, instruction_id: str, now: datetime, reason: str = "") -> Instruction:
        """企业退出响应：已分配容量立即恢复。"""
        instruction = self._require(instruction_id)
        self._ensure_active(instruction)
        updated = self._transition(
            instruction, InstructionStatus.EXITED, now, f"enterprise-exited:{reason}"
        )
        self._restore_capacity(updated, now, "enterprise-exit")
        return updated

    def dispatch_withdraw(self, instruction_id: str, now: datetime, reason: str = "") -> Instruction:
        """调度撤回指令：已分配容量立即恢复。"""
        instruction = self._require(instruction_id)
        self._ensure_active(instruction)
        updated = self._transition(
            instruction, InstructionStatus.WITHDRAWN, now, f"dispatch-withdrawn:{reason}"
        )
        self._restore_capacity(updated, now, "dispatch-withdraw")
        return updated

    def record_outcome(
        self, instruction_id: str, actual_shed_mw: float, now: datetime
    ) -> Instruction:
        """事后登记实际削峰；足额为履约，不足为部分履约。"""
        instruction = self._require(instruction_id)
        self._ensure_active(instruction)
        status = (
            InstructionStatus.FULFILLED
            if actual_shed_mw >= instruction.requested_reduction_mw
            else InstructionStatus.PARTIAL
        )
        updated = self._transition(instruction, status, now, "outcome-recorded")
        updated = replace(updated, actual_shed_mw=actual_shed_mw)
        self._instructions[instruction_id] = updated
        self._audit.append(
            (
                now,
                "outcome",
                {
                    "instruction_id": instruction_id,
                    "actual_shed_mw": actual_shed_mw,
                    "requested_mw": instruction.requested_reduction_mw,
                },
            )
        )
        return updated

    # ---- 台账与视图 ----

    def committed_mw(self, region: str, window: Window) -> float:
        """该地区该时窗仍处于活动状态的指令合计（退出/撤回即不再计入）。"""
        return sum(
            i.requested_reduction_mw
            for i in self._instructions.values()
            if i.region == region
            and i.window == window
            and i.status not in TERMINAL_STATUSES
        )

    def available_pool_mw(self, region: str, window: Window) -> float:
        """该地区该时窗仍可调配的响应容量。"""
        total = sum(e.max_reduction_mw for e in self._enterprises.values() if e.region == region)
        return total - self.committed_mw(region, window)

    def enterprise_view(self, enterprise_id: str) -> tuple[Instruction, ...]:
        """企业仅能看到自己的行动要求。"""
        return tuple(
            i for i in self._instructions.values() if i.enterprise_id == enterprise_id
        )

    def get(self, instruction_id: str) -> Instruction:
        return self._require(instruction_id)

    def instructions_for(self, region: str, window: Window) -> tuple[Instruction, ...]:
        return tuple(
            i
            for i in self._instructions.values()
            if i.region == region and i.window == window
        )

    def audit_log(self) -> tuple[tuple[datetime, str, dict], ...]:
        return tuple(self._audit)

    # ---- 内部 ----

    def _committed_by(self, enterprise_id: str, region: str, window: Window) -> float:
        return sum(
            i.requested_reduction_mw
            for i in self._instructions.values()
            if i.enterprise_id == enterprise_id
            and i.region == region
            and i.window == window
            and i.status not in TERMINAL_STATUSES
        )

    def _require(self, instruction_id: str) -> Instruction:
        if instruction_id not in self._instructions:
            raise ValueError(f"指令不存在: {instruction_id}")
        return self._instructions[instruction_id]

    @staticmethod
    def _ensure_active(instruction: Instruction) -> None:
        if instruction.status in TERMINAL_STATUSES:
            raise ValueError(
                f"指令 {instruction.id} 已处于终态 {instruction.status.value}，不可再变更"
            )

    def _transition(
        self,
        instruction: Instruction,
        status: InstructionStatus,
        now: datetime,
        event: str,
    ) -> Instruction:
        updated = replace(
            instruction, status=status, history=instruction.history + ((now, event),)
        )
        self._instructions[instruction.id] = updated
        self._audit.append(
            (now, event.split(":")[0], {"instruction_id": instruction.id, "event": event})
        )
        return updated

    def _restore_capacity(self, instruction: Instruction, now: datetime, cause: str) -> None:
        self._audit.append(
            (
                now,
                "capacity-restored",
                {
                    "instruction_id": instruction.id,
                    "enterprise_id": instruction.enterprise_id,
                    "window": instruction.window.key,
                    "restored_mw": instruction.requested_reduction_mw,
                    "cause": cause,
                },
            )
        )
