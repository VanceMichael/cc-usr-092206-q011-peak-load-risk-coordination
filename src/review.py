"""事后复盘：预测对照、履约结算、异常数据误报。

复盘在窗口/日结束后运行，直接对照：

* 预警时锁定的预测版本 vs 实际地区计量（regional_load）；
* 已分配容量 vs 企业关口计量算出的实际削峰、未履约容量；
* 触发预警的观测是否后来被标记为异常（anomalous），或实际备用其实
  充裕——两类都记为误报并给出原因，供改进计量接入与阈值。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime

from . import timewin
from .alerts import AlertBatch
from .forecast import ForecastVersion
from .ledger import Ledger, SeriesView, ObsFlag
from .response import ResponseCoordinator, WindowSettlement


@dataclass
class ForecastMiss:
    region: str
    window: str
    forecast_mw: float
    actual_mw: float
    error_mw: float
    transient_flagged: list[str]


@dataclass
class FalseAlert:
    alert_id: str
    region: str
    window: str
    level: str
    reason: str
    offending_obs: list[str]
    predicted_reserve_mw: float
    actual_reserve_mw: float


@dataclass
class ReviewReport:
    report_id: str
    generated_at: datetime
    forecast_version_id: str
    forecast_misses: list[ForecastMiss] = field(default_factory=list)
    settlements: list[WindowSettlement] = field(default_factory=list)
    false_alerts: list[FalseAlert] = field(default_factory=list)

    @property
    def total_allocated_mw(self) -> float:
        return round(sum(s.allocated_mw for s in self.settlements), 3)

    @property
    def total_actual_reduction_mw(self) -> float:
        return round(sum(s.actual_reduction_mw for s in self.settlements), 3)

    @property
    def total_unperformed_mw(self) -> float:
        return round(sum(s.unperformed_mw for s in self.settlements), 3)

    def summary(self) -> list[str]:
        lines = [
            f"复盘报告 {self.report_id}（依据预测版本 {self.forecast_version_id}）",
            f"分配容量合计 {self.total_allocated_mw:.1f} MW，"
            f"实际削峰 {self.total_actual_reduction_mw:.1f} MW，"
            f"未履约 {self.total_unperformed_mw:.1f} MW。",
        ]
        if self.false_alerts:
            lines.append(f"误报 {len(self.false_alerts)} 条：")
            for fa in self.false_alerts:
                lines.append(
                    f"  - {fa.alert_id} {fa.region} {fa.window}（{fa.level}）：{fa.reason}"
                    + (f"；涉数 {','.join(fa.offending_obs)}" if fa.offending_obs else ""))
        misses = sorted(self.forecast_misses,
                        key=lambda m: abs(m.error_mw), reverse=True)[:5]
        if misses:
            lines.append("偏差最大的时窗：")
            for m in misses:
                tag = "（含瞬时项，非趋势）" if m.transient_flagged else ""
                lines.append(
                    f"  - {m.region} {m.window}: 预测 {m.forecast_mw:.1f} / "
                    f"实际 {m.actual_mw:.1f}，偏差 {m.error_mw:+.1f} MW{tag}")
        return lines

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "forecast_version_id": self.forecast_version_id,
            "totals": {
                "allocated_mw": self.total_allocated_mw,
                "actual_reduction_mw": self.total_actual_reduction_mw,
                "unperformed_mw": self.total_unperformed_mw,
            },
            "forecast_misses": [asdict(m) for m in self.forecast_misses],
            "settlements": [s.to_dict() for s in self.settlements],
            "false_alerts": [asdict(f) for f in self.false_alerts],
        }


class Reviewer:
    def __init__(self, ledger: Ledger, coordinator: ResponseCoordinator,
                 capacity_mw: dict[str, float]):
        self.ledger = ledger
        self.coordinator = coordinator
        self.capacity = dict(capacity_mw)

    def run(self, report_id: str, batch: AlertBatch, fv: ForecastVersion,
            windows: list[datetime], generated_at: datetime) -> ReviewReport:
        view = self.ledger.view()
        report = ReviewReport(report_id, generated_at, fv.version_id)

        # 1) 预测 vs 实际
        actual_load = {}
        for region in fv.windows:
            s = view.series("regional_load", region=region)
            for w in windows:
                wf = fv.get(region, w)
                o = s.get(timewin.floor_to_window(w))
                if wf is None or o is None:
                    continue
                actual_load[(region, timewin.key(w))] = o.value
                report.forecast_misses.append(ForecastMiss(
                    region=region, window=wf.window,
                    forecast_mw=wf.forecast_mw, actual_mw=o.value,
                    error_mw=round(wf.forecast_mw - o.value, 3),
                    transient_flagged=list(wf.transient_flags)))

        # 2) 结算（企业关口计量）
        for w in windows:
            report.settlements.extend(
                self.coordinator.settle_window(w, view, generated_at))

        # 3) 误报判定
        anomalous_ids = {
            o.obs_id for o in self.ledger.all_observations()
            if ObsFlag.ANOMALOUS.value in o.flags
        }
        maint = {}
        for region in fv.windows:
            for w, o in view.series("maintenance", region=region).items():
                maint[(region, timewin.key(w))] = (
                    maint.get((region, timewin.key(w)), 0.0) + o.value)
        alerted_keys = {(a.region, a.window) for a in batch.alerts}
        for a in batch.alerts:
            offending = [oid for ids in a.source_obs.values() for oid in ids
                         if oid in anomalous_ids]
            actual = actual_load.get((a.region, a.window))
            actual_reserve = None
            if actual is not None:
                supply = self.capacity.get(a.region, 0.0) - maint.get(
                    (a.region, a.window), 0.0)
                actual_reserve = round(supply - actual, 3)
            if offending:
                report.false_alerts.append(FalseAlert(
                    a.alert_id, a.region, a.window, a.level,
                    "触发预警所用观测事后被标记为异常数据",
                    offending, a.reserve_mw,
                    actual_reserve if actual_reserve is not None else a.reserve_mw))
            elif actual_reserve is not None and actual_reserve > a.forecast_mw * 0.08:
                report.false_alerts.append(FalseAlert(
                    a.alert_id, a.region, a.window, a.level,
                    "实际备用率高于蓝色线，预测过度收紧",
                    [], a.reserve_mw, actual_reserve))
        return report
