"""日内预警引擎。

按当前预测版本逐窗口计算备用裕度：
    可用容量 = 装机容量 - 检修停运（按申报全额扣减）
    裕度比   = (可用容量 - 预测负荷) / 预测负荷
低于阈值即生成预警。每条预警附带逐项解释（数据项 + 来源观测 id）与假设，
分析人员可据此逐项核对预警依据。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from .models import (
    Alert,
    ExplanationItem,
    ForecastVersion,
    ObservationKind,
    Severity,
    Window,
)
from .observations import ObservationStore


@dataclass(frozen=True)
class CapacityProfile:
    region: str
    installed_mw: float


@dataclass(frozen=True)
class AlertThresholds:
    advisory_ratio: float = 0.15
    warning_ratio: float = 0.10
    critical_ratio: float = 0.05


_COMPONENT_LABELS = {
    "base": "基础负荷",
    "data_center": "数据中心",
    "high_end_manufacturing": "高端制造",
    "ev_charging": "充换电",
    "weather": "气象修正",
}


class AlertEngine:
    def __init__(
        self,
        store: ObservationStore,
        capacity: Mapping[str, CapacityProfile],
        thresholds: AlertThresholds | None = None,
    ) -> None:
        self._store = store
        self._capacity = dict(capacity)
        self._thresholds = thresholds or AlertThresholds()

    def available_capacity(self, region: str, window: Window) -> tuple[float, tuple[str, ...]]:
        """可用容量与所依据的检修观测 id。"""
        profile = self._capacity[region]
        outages = self._store.effective_observations(
            kind=ObservationKind.MAINTENANCE, region=region, window=window
        )
        outage_mw = sum(o.metrics.get("outage_mw", 0.0) for o in outages)
        return profile.installed_mw - outage_mw, tuple(o.id for o in outages)

    def evaluate(self, version: ForecastVersion, now: datetime) -> tuple[Alert, ...]:
        """对预测版本覆盖的全部时窗评估，返回达到阈值的预警。"""
        alerts: list[Alert] = []
        for (region, _key), result in sorted(
            version.results.items(), key=lambda item: (item[0][0], item[1].window.start)
        ):
            if region not in self._capacity:
                continue
            alert = self._evaluate_window(version, result.region, result.window, now)
            if alert is not None:
                alerts.append(alert)
        return tuple(alerts)

    def _evaluate_window(
        self, version: ForecastVersion, region: str, window: Window, now: datetime
    ) -> Alert | None:
        result = version.result_for(region, window)
        if result is None:
            return None
        available, maintenance_ids = self.available_capacity(region, window)
        predicted = result.predicted_load_mw
        margin = available - predicted
        ratio = margin / predicted if predicted > 0 else float("inf")
        severity = self._severity(ratio)
        if severity is None:
            return None

        explanation: list[ExplanationItem] = [
            ExplanationItem(
                label="预测负荷",
                value=f"{predicted:.1f} MW（预测版本 v{version.version}）",
                source_observation_ids=result.input_observation_ids,
            )
        ]
        for name, value in result.components.items():
            label = _COMPONENT_LABELS.get(name, name)
            explanation.append(
                ExplanationItem(
                    label=f"分量·{label}",
                    value=f"{value:+.1f} MW",
                    source_observation_ids=result.component_inputs.get(name, ()),
                )
            )
        profile = self._capacity[region]
        outage_mw = profile.installed_mw - available
        explanation.append(
            ExplanationItem(
                label="可用容量",
                value=f"{available:.1f} MW = 装机 {profile.installed_mw:.1f} - 检修停运 {outage_mw:.1f}",
                source_observation_ids=maintenance_ids,
            )
        )
        explanation.append(
            ExplanationItem(
                label="备用裕度",
                value=(
                    f"{margin:.1f} MW（{ratio:.1%}）；阈值："
                    f"提示<{self._thresholds.advisory_ratio:.0%} "
                    f"预警<{self._thresholds.warning_ratio:.0%} "
                    f"严重<{self._thresholds.critical_ratio:.0%}"
                ),
            )
        )
        assumptions = result.assumptions + ("检修计划按申报停运容量全额扣减",)
        alert_id = f"AL-v{version.version}-{region}-{window.start:%Y%m%d%H}"
        return Alert(
            id=alert_id,
            region=region,
            window=window,
            severity=severity,
            forecast_version=version.version,
            predicted_load_mw=predicted,
            available_capacity_mw=available,
            reserve_margin_mw=margin,
            reserve_margin_ratio=ratio,
            explanation=tuple(explanation),
            assumptions=assumptions,
            created_at=now,
        )

    def severity_for(self, ratio: float) -> Severity | None:
        """按裕度比判定预警等级；不低于提示阈值时返回 None。"""
        if ratio < self._thresholds.critical_ratio:
            return Severity.CRITICAL
        if ratio < self._thresholds.warning_ratio:
            return Severity.WARNING
        if ratio < self._thresholds.advisory_ratio:
            return Severity.ADVISORY
        return None

    def _severity(self, ratio: float) -> Severity | None:
        return self.severity_for(ratio)


def explain_alert(alert: Alert) -> dict:
    """把预警解释展开为分析人员可逐项核对的结构。"""
    return {
        "alert_id": alert.id,
        "region": alert.region,
        "window": alert.window.key,
        "severity": alert.severity.value,
        "forecast_version": alert.forecast_version,
        "items": [
            {
                "label": item.label,
                "value": item.value,
                "sources": list(item.source_observation_ids),
            }
            for item in alert.explanation
        ],
        "assumptions": list(alert.assumptions),
    }
