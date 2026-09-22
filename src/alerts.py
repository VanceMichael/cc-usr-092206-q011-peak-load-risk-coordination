"""局部备用容量预警与逐项解释。

调度员在月报形成前最关心两件事：

1. 数据中心、充换电、高端制造的增长是否把某个地区的备用容量挤到红线以下；
2. 看到的负荷变化里，哪些只是台风降雨造成的居民短期下降（瞬时项），
   不能当成长周期趋势写进月报。

预警全部基于某个冻结的预测版本生成，自身也按批次只追加保存；
每条预警都带：贡献分解、所用观测 obs_id、生效假设，可逐项解释。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from . import timewin
from .forecast import ForecastVersion, WindowForecast
from .ledger import SeriesView


@dataclass(frozen=True)
class ReservePolicy:
    """备用容量阈值策略（可按地区覆盖）。"""
    # 以负荷比例表达的备用率阈值：蓝/橙/红
    blue_reserve_ratio: float = 0.08
    orange_reserve_ratio: float = 0.05
    red_reserve_ratio: float = 0.02
    region_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def ratios_for(self, region: str) -> tuple[float, float, float]:
        ov = self.region_overrides.get(region, {})
        return (
            ov.get("blue", self.blue_reserve_ratio),
            ov.get("orange", self.orange_reserve_ratio),
            ov.get("red", self.red_reserve_ratio),
        )


LEVELS = ("red", "orange", "blue", "normal")


@dataclass
class Alert:
    alert_id: str
    region: str
    window: str
    level: str                          # red / orange / blue
    forecast_mw: float
    supply_mw: float
    reserve_mw: float
    reserve_ratio: float
    gap_mw: float                       # 恢复到蓝色（最低合格）线还缺的容量
    growth_pressure_mw: float           # 三类长周期增长合计
    transient_mw: float                 # 台风居民短期下降等瞬时项合计
    transient_flags: list[str]
    forecast_version: str
    source_obs: dict[str, list[str]]
    assumptions: dict
    explanation: list[str]
    maintenance_derate_mw: float
    status: str = "active"              # active / withdrawn / superseded
    issued_at: datetime | None = None

    def explain(self) -> list[dict]:
        """分析人员视角的逐项解释：每项数据/假设一条。"""
        items = [
            {"item": "预测负荷", "value_mw": self.forecast_mw,
             "basis": f"预测版本 {self.forecast_version}",
             "obs_ids": self.source_obs.get("baseline", [])},
            {"item": "供电能力", "value_mw": self.supply_mw,
             "basis": f"地区供电能力扣减检修 {self.maintenance_derate_mw} MW 后的净值"},
            {"item": "备用容量", "value_mw": self.reserve_mw,
             "basis": f"供电能力 - 预测负荷；备用率 {self.reserve_ratio:.2%}"},
            {"item": "长周期增长压力", "value_mw": self.growth_pressure_mw,
             "basis": "数据中心 + 充换电 + 高端制造年化增长折算",
             "obs_ids": self.source_obs.get("industry_load", [])},
            {"item": "充换电预约增量",
             "value_mw": _contrib(self, "ev_booking"),
             "obs_ids": self.source_obs.get("ev_demand", [])},
            {"item": "高温制冷调整",
             "value_mw": _contrib(self, "weather_cooling"),
             "obs_ids": self.source_obs.get("weather", [])},
            {"item": "台风居民短期下降",
             "value_mw": self.transient_mw,
             "basis": "瞬时气象项，已与长周期增长分列，不进入趋势判断",
             "obs_ids": (self.source_obs.get("weather", [])
                         + self.source_obs.get("residential", [])),
             "transient": True},
            {"item": "预警参数", "basis": "备用率阈值（蓝/橙/红）",
             "params": {
                 "ratios": self.assumptions.get("reserve_ratios"),
                 "forecast": {
                     k: v for k, v in self.assumptions.items()
                     if k != "reserve_ratios"}}},
            {"item": "缺口", "value_mw": self.gap_mw,
             "basis": "恢复到蓝色备用线所需的需求响应容量"},
        ]
        return items

    def to_dict(self) -> dict:
        d = asdict(self)
        d["issued_at"] = self.issued_at.isoformat() if self.issued_at else None
        return d

    @staticmethod
    def from_dict(d: dict) -> "Alert":
        d = dict(d)
        d["issued_at"] = (datetime.fromisoformat(d["issued_at"])
                          if d.get("issued_at") else None)
        return Alert(**d)


def _contrib(alert: Alert, key: str) -> float:
    # WindowForecast 的贡献明细在生成时已展开到 source 数据，这里从
    # explanation 生成器读不到 contributions，故存于 assumptions 快照：
    return float(alert.assumptions.get("contributions", {}).get(key, 0.0))


@dataclass
class AlertBatch:
    """一次预警评估的不可变批次（日内可多次发布）。"""
    batch_id: str
    created_at: datetime
    forecast_version_id: str
    alerts: list[Alert]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "batch_id": self.batch_id,
            "created_at": self.created_at.isoformat(),
            "forecast_version_id": self.forecast_version_id,
            "alerts": [a.to_dict() for a in self.alerts],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "AlertBatch":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return AlertBatch(
            batch_id=d["batch_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            forecast_version_id=d["forecast_version_id"],
            alerts=[Alert.from_dict(a) for a in d["alerts"]],
        )


class AlertEngine:
    def __init__(self, capacity_mw: dict[str, float],
                 policy: ReservePolicy | None = None):
        self.capacity = dict(capacity_mw)
        self.policy = policy or ReservePolicy()

    def evaluate(
        self,
        batch_id: str,
        fv: ForecastVersion,
        view: SeriesView,
        horizon: list[datetime],
        created_at: datetime,
    ) -> AlertBatch:
        alerts: list[Alert] = []
        seq = 0
        scoped = view._ledger.view(as_of=created_at)
        for region in sorted(fv.windows):
            blue, orange, red = self.policy.ratios_for(region)
            maint = scoped.series("maintenance", region=region)
            for w in horizon:
                wf = fv.get(region, w)
                if wf is None:
                    continue
                derate = round(sum(
                    o.value for ww, o in maint.items() if ww == timewin.floor_to_window(w)
                ), 3)
                supply = round(self.capacity.get(region, 0.0) - derate, 3)
                reserve = round(supply - wf.forecast_mw, 3)
                ratio = reserve / wf.forecast_mw if wf.forecast_mw > 0 else 1.0
                level = ("red" if ratio < red else
                         "orange" if ratio < orange else
                         "blue" if ratio < blue else "normal")
                if level == "normal":
                    continue
                seq += 1
                growth = round(sum(
                    wf.contributions_mw.get(f"{s}_growth", 0.0)
                    for s in ("data_center", "ev_charge", "high_end_manufacturing")
                ), 3)
                transient = round(sum(
                    v for k, v in wf.contributions_mw.items()
                    if k.startswith("typhoon_")), 3)
                gap = round(wf.forecast_mw * blue - reserve, 3)
                assumed = dict(wf.assumed)
                assumed["reserve_ratios"] = {"blue": blue, "orange": orange, "red": red}
                assumed["contributions"] = dict(wf.contributions_mw)
                explanation = self._explain(
                    region, level, wf, supply, reserve, ratio, gap, growth,
                    transient, derate, blue)
                alerts.append(Alert(
                    alert_id=f"{batch_id}-A{seq:03d}",
                    region=region,
                    window=wf.window,
                    level=level,
                    forecast_mw=wf.forecast_mw,
                    supply_mw=supply,
                    reserve_mw=reserve,
                    reserve_ratio=round(ratio, 5),
                    gap_mw=max(gap, 0.0),
                    growth_pressure_mw=growth,
                    transient_mw=transient,
                    transient_flags=list(wf.transient_flags),
                    forecast_version=fv.version_id,
                    source_obs={k: list(v) for k, v in wf.source_obs.items()},
                    assumptions=assumed,
                    explanation=explanation,
                    maintenance_derate_mw=derate,
                    issued_at=created_at,
                ))
        return AlertBatch(batch_id, created_at, fv.version_id, alerts)

    def _explain(self, region, level, wf: WindowForecast, supply, reserve,
                 ratio, gap, growth, transient, derate, blue) -> list[str]:
        lines = [
            f"{region} {wf.window} 预测负荷 {wf.forecast_mw:.1f} MW，"
            f"供电能力（扣检修 {derate:.1f} MW）{supply:.1f} MW，"
            f"备用 {reserve:.1f} MW（备用率 {ratio:.2%}），触发{_lv(level)}预警。",
            f"长周期增长压力合计 {growth:.1f} MW："
            + "、".join(f"{_sec_name(k)} {v:+.1f} MW"
                       for k, v in wf.contributions_mw.items()
                       if k.endswith("_growth"))
            + "；这些是判断增长是否挤压备用的依据。",
            f"若要恢复到 {blue:.0%} 备用线，需需求响应约 {max(gap, 0):.1f} MW。",
        ]
        if wf.transient_flags:
            lines.append(
                f"含瞬时项 {transient:.1f} MW（{','.join(wf.transient_flags)}）："
                "台风降雨导致的居民短期用电下降，仅影响本时窗附近，"
                "不得作为长周期趋势写入月报或外推到跨日峰谷。")
        return lines


def _lv(level: str) -> str:
    return {"red": "红色", "orange": "橙色", "blue": "蓝色"}[level]


def _sec_name(key: str) -> str:
    return {
        "data_center_growth": "数据中心",
        "ev_charge_growth": "充换电",
        "high_end_manufacturing_growth": "高端制造",
    }.get(key, key)
