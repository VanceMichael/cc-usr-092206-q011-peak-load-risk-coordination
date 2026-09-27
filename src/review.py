"""事后复盘。

按 (地区, 时窗) 对照：
- 预测负荷 vs 实际有效计量（更正后的最新值；原始观测仍保留在库中）；
- 实际削峰 vs 承诺容量，得出未履约容量（企业退出、部分履约、未登记结果均计入）；
- 预警是否误报：若按实际负荷计算备用裕度本不触发预警，则为误报，
  并归因到参与预测的异常观测（质量标记异常或事后被更正的读数）。
"""

from __future__ import annotations

from datetime import datetime

from .models import (
    Alert,
    ForecastVersion,
    InstructionStatus,
    ObservationKind,
    Quality,
    ReviewReport,
    Window,
    WindowReview,
)
from .observations import ObservationStore
from .response import ResponseCoordinator
from .risks import AlertEngine


class ReviewEngine:
    def __init__(self, store: ObservationStore, alerts_engine: AlertEngine) -> None:
        self._store = store
        self._alerts_engine = alerts_engine

    def review(
        self,
        scope: tuple[tuple[str, Window], ...],
        version: ForecastVersion,
        alerts: tuple[Alert, ...],
        coordinator: ResponseCoordinator,
        now: datetime,
    ) -> ReviewReport:
        rows: list[WindowReview] = []
        for region, window in scope:
            rows.append(self._review_window(region, window, version, alerts, coordinator))
        return ReviewReport(
            generated_at=now,
            rows=tuple(rows),
            total_committed_mw=sum(r.committed_mw for r in rows),
            total_actual_shed_mw=sum(r.actual_shed_mw for r in rows),
            total_unfulfilled_mw=sum(r.unfulfilled_mw for r in rows),
            alert_count=sum(1 for r in rows if r.alert_issued),
            false_alarm_count=sum(1 for r in rows if r.false_alarm),
        )

    def _review_window(
        self,
        region: str,
        window: Window,
        version: ForecastVersion,
        alerts: tuple[Alert, ...],
        coordinator: ResponseCoordinator,
    ) -> WindowReview:
        result = version.result_for(region, window)
        forecast_mw = result.predicted_load_mw if result else None

        actual_mw = self._actual_load(region, window)
        error = (forecast_mw - actual_mw) if (forecast_mw is not None and actual_mw is not None) else None

        window_alerts = [a for a in alerts if a.region == region and a.window == window]
        alert_issued = len(window_alerts) > 0

        actual_ok: bool | None = None
        if actual_mw is not None:
            available, _ = self._alerts_engine.available_capacity(region, window)
            ratio = (available - actual_mw) / actual_mw if actual_mw > 0 else float("inf")
            actual_ok = self._alerts_engine.severity_for(ratio) is None

        false_alarm = alert_issued and actual_ok is True
        causes = self._false_alarm_causes(result.input_observation_ids) if (false_alarm and result) else ()

        committed = 0.0
        shed = 0.0
        for instruction in coordinator.instructions_for(region, window):
            if instruction.status is InstructionStatus.WITHDRAWN:
                continue  # 调度撤回不计入企业承诺
            committed += instruction.requested_reduction_mw
            if instruction.actual_shed_mw is not None:
                shed += instruction.actual_shed_mw
        unfulfilled = committed - shed

        return WindowReview(
            region=region,
            window=window,
            forecast_mw=forecast_mw,
            actual_mw=actual_mw,
            forecast_error_mw=error,
            alert_issued=alert_issued,
            actual_reserve_ok=actual_ok,
            false_alarm=false_alarm,
            false_alarm_causes=tuple(causes),
            committed_mw=committed,
            actual_shed_mw=shed,
            unfulfilled_mw=unfulfilled,
        )

    def _actual_load(self, region: str, window: Window) -> float | None:
        readings = [
            obs
            for obs in self._store.effective_observations(
                kind=ObservationKind.METER, region=region, window=window
            )
            if obs.industry is None and "load_mw" in obs.metrics
        ]
        if not readings:
            return None
        latest = max(readings, key=lambda o: o.ingested_at)
        return latest.metrics["load_mw"]

    def _false_alarm_causes(self, input_ids: tuple[str, ...]) -> list[str]:
        causes: list[str] = []
        for oid in input_ids:
            obs = self._store.get(oid)
            if obs.quality is not Quality.NORMAL:
                causes.append(f"观测 {oid} 质量标记为 {obs.quality.value}")
            if self._store.is_superseded(oid):
                causes.append(
                    f"观测 {oid} 事后被更正为 {self._store.effective_id(oid)}"
                )
        return causes
