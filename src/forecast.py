"""版本化负荷预测与增量重算。

模型刻意保持透明可解释（调度员要能逐项核对）：

    预测负荷 = 基准曲线
             + 数据中心增长 + 充换电增长 + 高端制造增长
             + 高温制冷调整
             - 台风降雨导致的居民短期下降
             + 充换电预约需求

关键规则：

* 预测结果只以不可变版本（ForecastVersion）保存，新版本永不覆盖旧版本。
* 迟到读数、计量更正或人工触发的新预测，只重算“受影响时窗”
  （affected_windows）；其余时窗从上一版本原样沿用，并在版本里记录
  carried_from。月报形成前的任意时点都能回看当时用的是哪一版预测。
* 每个时窗结果记录所用观测 obs_id 与当时假设，预警解释直接引用。
* 台风降雨造成的居民用电下降被显式标记为 weather_transient，
  与长周期行业增长分开，避免把短期扰动当成趋势。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path

from . import timewin
from .ledger import SeriesView
GROWTH_SECTORS = ("data_center", "ev_charge", "high_end_manufacturing")


@dataclass(frozen=True)
class ForecastAssumptions:
    """预测假设的快照（每个版本冻结一份，解释预警时直接展示）。"""
    name: str = "transparent-v1"
    # 行业年化增长率，按行业计量基线折算到时窗增量
    annual_growth: dict[str, float] = field(default_factory=lambda: {
        "data_center": 0.15,
        "ev_charge": 0.30,
        "high_end_manufacturing": 0.08,
    })
    # 高温制冷：气温超过 cooling_setpoint_c 后，每升高 1°C 的负荷系数
    # （作用于基准负荷的比例）
    cooling_setpoint_c: float = 30.0
    cooling_coef_per_c: float = 0.004
    # 台风降雨：1 小时降雨≥阈值且风速≥阈值时，居民负荷按比例短期下降
    typhoon_rain_mm: float = 20.0
    typhoon_wind_kmh: float = 60.0
    typhoon_residential_drop: float = 0.12
    # 基准曲线回看天数（取同一时刻历史均值）
    baseline_lookback_days: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ForecastAssumptions":
        return ForecastAssumptions(**d)


@dataclass
class WindowForecast:
    region: str
    window: str                       # ISO 时窗起点
    forecast_mw: float
    contributions_mw: dict[str, float]   # 各项贡献，正负号带方向
    transient_flags: list[str]        # 如 ["typhoon_residential_drop"]
    source_obs: dict[str, list[str]]  # 用途 -> obs_id 列表
    assumed: dict                     # 实际生效的关键参数值
    recomputed: bool = True
    carried_from: str | None = None   # 沿用自哪个版本

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "WindowForecast":
        return WindowForecast(**d)


@dataclass
class ForecastVersion:
    version_id: str
    created_at: datetime
    assumptions: ForecastAssumptions
    windows: dict[str, dict[str, WindowForecast]]  # region -> wkey -> 结果
    trigger: str                          # late_reading / correction / manual / scheduled
    affected_windows: list[str]           # 本次重算的时窗键
    parent_version: str | None = None
    note: str = ""

    def get(self, region: str, window: str | datetime) -> WindowForecast | None:
        return self.windows.get(region, {}).get(timewin.key(
            window if isinstance(window, datetime) else datetime.fromisoformat(window)))

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at.isoformat(),
            "assumptions": self.assumptions.to_dict(),
            "trigger": self.trigger,
            "affected_windows": list(self.affected_windows),
            "parent_version": self.parent_version,
            "note": self.note,
            "windows": {
                region: {wk: wf.to_dict() for wk, wf in m.items()}
                for region, m in self.windows.items()
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "ForecastVersion":
        return ForecastVersion(
            version_id=d["version_id"],
            created_at=datetime.fromisoformat(d["created_at"]),
            assumptions=ForecastAssumptions.from_dict(d["assumptions"]),
            trigger=d["trigger"],
            affected_windows=list(d["affected_windows"]),
            parent_version=d.get("parent_version"),
            note=d.get("note", ""),
            windows={
                region: {wk: WindowForecast.from_dict(wf) for wk, wf in m.items()}
                for region, m in d["windows"].items()
            },
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "ForecastVersion":
        return ForecastVersion.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8")))


class Forecaster:
    """依据台账视图与冻结假设生成预测版本。"""

    def __init__(self, assumptions: ForecastAssumptions | None = None):
        self.assumptions = assumptions or ForecastAssumptions()

    def affected_windows_for(self, obs_ids: list[str], view: SeriesView,
                             horizon: list[datetime]) -> list[str]:
        """根据新增/更正/标记的观测，推算受影响时窗。

        * 气象观测按天气惯性向前影响 2 个时窗（半小时），保守地避免把
          降雨效应扩散成“长趋势”；
        * 地区计量通过持续项向后影响 2 个时窗（最近计量水平对后续刻钟
          的惯性外推）；
        * 行业计量、检修、充换电预约只影响其自身时窗。
        """
        span_by_kind = {"weather": 2, "regional_load": 2}
        affected: set[str] = set()
        horizon_keys = {timewin.key(w) for w in horizon}
        all_obs = {o.obs_id: o for o in view._ledger.all_observations()}
        for oid in obs_ids:
            o = all_obs.get(oid)
            if o is None:
                continue
            span = span_by_kind.get(o.kind, 1)
            for k in range(span):
                wk = timewin.key(o.window + timedelta(minutes=15 * k))
                if wk in horizon_keys:
                    affected.add(wk)
        return sorted(affected)

    def run(
        self,
        version_id: str,
        view: SeriesView,
        horizon: list[datetime],
        regions: list[str],
        created_at: datetime,
        trigger: str = "manual",
        affected_windows: list[str] | None = None,
        parent: ForecastVersion | None = None,
        static_baseline: dict[str, dict[str, float]] | None = None,
        note: str = "",
    ) -> ForecastVersion:
        """生成新版本。affected_windows 之外的时窗从 parent 原样沿用。"""
        affected_set = set(affected_windows or [timewin.key(w) for w in horizon])
        # 严格按版本创建时刻取数：未来才到的读数不能影响历史版本
        scoped = view._ledger.view(as_of=created_at)
        result: dict[str, dict[str, WindowForecast]] = {}
        for region in regions:
            result[region] = {}
            for w in horizon:
                wk = timewin.key(w)
                if wk not in affected_set and parent is not None:
                    prior = parent.get(region, w)
                    if prior is not None:
                        carried = WindowForecast.from_dict(prior.to_dict())
                        carried.recomputed = False
                        carried.carried_from = parent.version_id
                        result[region][wk] = carried
                        continue
                result[region][wk] = self._compute_window(region, w, scoped,
                                                           static_baseline or {})
        return ForecastVersion(
            version_id=version_id,
            created_at=created_at,
            assumptions=self.assumptions,
            windows=result,
            trigger=trigger,
            affected_windows=sorted(affected_set),
            parent_version=parent.version_id if parent else None,
            note=note,
        )

    # ---- 单时窗透明计算 -------------------------------------------------

    def _compute_window(self, region: str, w: datetime, view: SeriesView,
                        static_baseline: dict) -> WindowForecast:
        a = self.assumptions
        contrib: dict[str, float] = {}
        sources: dict[str, list[str]] = {}

        base, base_ids = self._baseline(region, w, view, static_baseline)
        contrib["baseline"] = round(base, 3)
        sources["baseline"] = base_ids

        # 计量水平持续项：最近 45 分钟内地区实测相对同时刻规划基准的偏移，
        # 按惯性进入后续时窗。迟到读数由此影响相邻时窗，更远时窗不被重算，
        # 也不会回写跨日峰谷。注意计入的是“偏移”而非实测全量，避免重复。
        persist, persist_ids = self._persistence(region, w, view,
                                                 static_baseline)
        contrib["level_persistence"] = persist
        if persist_ids:
            sources["regional_load"] = persist_ids

        # 行业增长：基准曲线取规划/去年典型日口径时，当前行业计量 × 年化
        # 增长率即“同比新增需求”——这正是判断增长是否挤压备用的边际量。
        for sec in GROWTH_SECTORS:
            obs = view.at("industry_load", w, region=region, sector=sec)
            if obs:
                rate = a.annual_growth[sec]
                contrib[f"{sec}_growth"] = round(obs.value * rate, 3)
                sources.setdefault("industry_load", []).append(obs.obs_id)

        # 充换电预约需求（除行业计量增长外的可调度增量）
        ev = view.at("ev_demand", w, region=region)
        if ev:
            contrib["ev_booking"] = round(ev.value, 3)
            sources["ev_demand"] = [ev.obs_id]

        transient: list[str] = []
        weather = view.at("weather", w, region=region)
        if weather:
            sources["weather"] = [weather.obs_id]
            temp = weather.metrics.get("temp_c", 0.0)
            rain = weather.metrics.get("rainfall_mm", 0.0)
            wind = weather.metrics.get("wind_km_h", 0.0)
            if temp > a.cooling_setpoint_c:
                contrib["weather_cooling"] = round(
                    base * a.cooling_coef_per_c * (temp - a.cooling_setpoint_c), 3)
            else:
                contrib["weather_cooling"] = 0.0
            if rain >= a.typhoon_rain_mm and wind >= a.typhoon_wind_kmh:
                res = view.series("industry_load", region=region, sector="residential")
                # 居民短期下降：以最近的居民计量估计
                recent_res = max(
                    (o for ww, o in res.items() if ww <= w),
                    key=lambda o: o.window, default=None)
                drop = 0.0
                if recent_res:
                    drop = round(recent_res.value * a.typhoon_residential_drop, 3)
                    sources.setdefault("residential", []).append(recent_res.obs_id)
                contrib["typhoon_residential_drop"] = -drop
                transient.append("typhoon_residential_drop")

        total = round(sum(contrib.values()), 3)
        return WindowForecast(
            region=region,
            window=timewin.key(w),
            forecast_mw=max(total, 0.0),
            contributions_mw=contrib,
            transient_flags=transient,
            source_obs=sources,
            assumed={
                "annual_growth": dict(a.annual_growth),
                "cooling_setpoint_c": a.cooling_setpoint_c,
                "cooling_coef_per_c": a.cooling_coef_per_c,
                "typhoon_rain_mm": a.typhoon_rain_mm,
                "typhoon_wind_kmh": a.typhoon_wind_kmh,
                "typhoon_residential_drop": a.typhoon_residential_drop,
            },
        )

    def _baseline(self, region: str, w: datetime, view: SeriesView,
                  static_baseline: dict) -> tuple[float, list[str]]:
        """基准曲线，优先级：前 1~7 天同时刻实测均值 > 当日规划基准
        （planned_load，须在时窗开始前下发）> 静态基准表。
        """
        load_hist = view.series("regional_load", region=region)
        vals, ids = [], []
        for d in range(1, self.assumptions.baseline_lookback_days + 1):
            cand = w - timedelta(days=d)
            o = load_hist.get(timewin.floor_to_window(cand))
            if o:
                vals.append(o.value)
                ids.append(o.obs_id)
        if vals:
            return sum(vals) / len(vals), ids
        planned = view.series("planned_load", region=region).get(
            timewin.floor_to_window(w))
        if planned is not None and planned.received_at <= w:
            return planned.value, [planned.obs_id]
        tbl = static_baseline.get(region, {})
        return float(tbl.get(timewin.key(w), 0.0)), []

    def _planned_value(self, region: str, w: datetime, view: SeriesView,
                       static_baseline: dict) -> tuple[float, list[str]]:
        planned = view.series("planned_load", region=region).get(
            timewin.floor_to_window(w))
        if planned is not None:
            return planned.value, [planned.obs_id]
        tbl = static_baseline.get(region, {})
        v = tbl.get(timewin.key(w))
        return (float(v), []) if v is not None else (0.0, [])

    def _persistence(self, region: str, w: datetime, view: SeriesView,
                     static_baseline: dict) -> tuple[float, list[str]]:
        """最近 45 分钟内最新到达实测相对规划基准的偏移（惯性持续项）。"""
        hist = view.series("regional_load", region=region)
        candidates = [
            o for ww, o in hist.items()
            if w - timedelta(minutes=45) <= ww <= w
        ]
        if not candidates:
            return 0.0, []
        # 按接收时间选最新到达的计量（迟到读数正是因此传到后续时窗）
        latest = max(candidates, key=lambda o: (o.received_at, o.window))
        planned, ids = self._planned_value(region, latest.window, view,
                                           static_baseline)
        return round(latest.value - planned, 3), [latest.obs_id] + ids
