"""负荷风险协调平台门面。

串联：数据接入 → 版本化预测 → 日内预警 → 需求响应 → 事后复盘。
- 接入的迟到读数/计量更正累积为待重算时窗，重算只覆盖这些时窗；
- 设备检修不影响负荷预测，仅在下一次预警评估中体现为可用容量变化；
- 企业视图与内部视图分离，参与企业仅能看到自己的行动要求。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .forecasts import ForecastConfig, ForecastEngine
from .models import (
    Alert,
    Enterprise,
    ForecastVersion,
    Instruction,
    Observation,
    ObservationKind,
    ReviewReport,
    Window,
)
from .observations import ObservationStore
from .response import ResponseCoordinator
from .review import ReviewEngine
from .risks import AlertEngine, AlertThresholds, CapacityProfile, explain_alert

# 会影响负荷预测的观测类型；检修只影响可用容量，不触发预测重算
_FORECAST_KINDS = frozenset(
    {ObservationKind.METER, ObservationKind.WEATHER, ObservationKind.EV_DEMAND}
)


@dataclass(frozen=True)
class IngestResult:
    observation: Observation
    affected_keys: tuple[tuple[str, Window], ...]
    triggers_forecast_recompute: bool


class LoadRiskPlatform:
    def __init__(
        self,
        forecast_config: ForecastConfig,
        capacity: tuple[CapacityProfile, ...],
        enterprises: tuple[Enterprise, ...],
        thresholds: AlertThresholds | None = None,
    ) -> None:
        self.store = ObservationStore()
        self.forecasts = ForecastEngine(self.store, forecast_config)
        self.alerts_engine = AlertEngine(
            self.store, {c.region: c for c in capacity}, thresholds
        )
        self.responses = ResponseCoordinator(enterprises)
        self.reviews = ReviewEngine(self.store, self.alerts_engine)

        self._regions: tuple[str, ...] = ()
        self._windows: tuple[Window, ...] = ()
        self._pending_keys: set[tuple[str, Window]] = set()
        self._pending_correction = False
        self._active_alerts: dict[tuple[str, str], Alert] = {}
        self._alert_history: list[Alert] = []

    # ---- 数据接入 ----

    def set_horizon(self, regions: tuple[str, ...], windows: tuple[Window, ...]) -> None:
        self._regions = regions
        self._windows = windows

    def ingest(self, obs: Observation) -> IngestResult:
        self.store.append(obs)
        triggers = obs.kind in _FORECAST_KINDS
        keys = (
            self.forecasts.affected_windows(obs, self._windows) if triggers else frozenset()
        )
        if keys:
            self._pending_keys.update(keys)
            if obs.corrects is not None:
                self._pending_correction = True
        return IngestResult(
            observation=obs,
            affected_keys=tuple(sorted(keys, key=lambda k: (k[0], k[1].start))),
            triggers_forecast_recompute=bool(keys),
        )

    # ---- 预测（按版本保存，局部重算） ----

    def run_forecast(self, now: datetime, reason: str = "initial") -> ForecastVersion:
        version = self.forecasts.initial_forecast(self._regions, self._windows, now, reason)
        self._pending_keys.clear()
        self._pending_correction = False
        return version

    def recompute_pending(self, now: datetime) -> ForecastVersion | None:
        """只重算自上次预测以来受迟到读数/更正影响的时窗；无待办时返回 None。"""
        if not self._pending_keys:
            return None
        reason = "correction" if self._pending_correction else "late-reading"
        keys = frozenset(self._pending_keys)
        self._pending_keys.clear()
        self._pending_correction = False
        return self.forecasts.recompute(keys, now, reason)

    def request_new_forecast(
        self, keys: frozenset[tuple[str, Window]], now: datetime
    ) -> ForecastVersion:
        """新预测请求：只重算指定时窗，其余自上一版本结转。"""
        return self.forecasts.recompute(keys, now, "new-forecast")

    # ---- 日内预警 ----

    def evaluate_alerts(self, now: datetime) -> tuple[Alert, ...]:
        """按当前预测版本评估；同一时窗的新预警取代旧预警（旧预警留档）。"""
        version = self.forecasts.current
        if version is None:
            raise ValueError("尚无预测版本，请先执行 run_forecast")
        fresh = self.alerts_engine.evaluate(version, now)
        fresh_by_key = {(a.region, a.window.key): a for a in fresh}
        for key, old in self._active_alerts.items():
            if key not in fresh_by_key or fresh_by_key[key].id != old.id:
                self._alert_history.append(old)
        self._active_alerts = fresh_by_key
        return fresh

    @property
    def active_alerts(self) -> tuple[Alert, ...]:
        return tuple(self._active_alerts.values())

    def get_alert(self, alert_id: str) -> Alert:
        for alert in list(self._active_alerts.values()) + self._alert_history:
            if alert.id == alert_id:
                return alert
        raise ValueError(f"预警不存在: {alert_id}")

    def explain_alert(self, alert_id: str) -> dict:
        """分析人员逐项核对预警所用的数据与假设。"""
        return explain_alert(self.get_alert(alert_id))

    # ---- 需求响应 ----

    def plan_response(self, alert_id: str, now: datetime) -> tuple[Instruction, ...]:
        return self.responses.plan(self.get_alert(alert_id), now)

    def enterprise_view(self, enterprise_id: str) -> tuple[Instruction, ...]:
        """参与企业仅收到自己的行动要求。"""
        return self.responses.enterprise_view(enterprise_id)

    # ---- 事后复盘 ----

    def review(
        self,
        now: datetime,
        scope: tuple[tuple[str, Window], ...] | None = None,
    ) -> ReviewReport:
        version = self.forecasts.current
        if version is None:
            raise ValueError("尚无预测版本，无法复盘")
        if scope is None:
            scope = tuple(
                (region, window) for region in self._regions for window in self._windows
            )
        all_alerts = tuple(self._active_alerts.values()) + tuple(self._alert_history)
        return self.reviews.review(scope, version, all_alerts, self.responses, now)
