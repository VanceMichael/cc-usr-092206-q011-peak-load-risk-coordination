"""版本化增量预测引擎。

- 每次计算产生一个完整快照版本（ForecastVersion），按版本保存、可回溯。
- 迟到读数、计量更正或新预测请求只重算受影响时窗；未受影响窗口的结果
  自上一版本原样结转（同一对象），版本仍保持完整。
- 台风/强降雨造成的居民用电下降按短时扰动处理，只修正受影响时窗，
  不计入长期趋势；该假设随窗口结果保存，供预警解释逐项核对。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

from .models import (
    INDUSTRY_DATA_CENTER,
    INDUSTRY_HIGH_END_MFG,
    ForecastVersion,
    ForecastWindowResult,
    Observation,
    ObservationKind,
    Window,
)
from .observations import ObservationStore


@dataclass(frozen=True)
class ForecastConfig:
    default_base_mw: Mapping[str, float]                 # 地区 -> 无历史时的兜底基础负荷
    industry_base_mw: Mapping[tuple[str, str], float]    # (地区, 行业) -> 增长外推基线
    industry_growth_daily: Mapping[str, float]           # 行业 -> 日增速（如 0.002）
    growth_reference_date: date                          # 基线日期
    residential_share: float = 0.30                      # 居民用电占基础负荷比例
    rain_dip_ratio: float = 0.20                         # 强降雨时居民负荷的短时下降比例
    rain_threshold_mm: float = 10.0
    heat_sensitivity_mw_per_c: float = 25.0              # 超过参考温度后每度增加的负荷
    reference_temp_c: float = 33.0


TRACKED_INDUSTRIES = (INDUSTRY_DATA_CENTER, INDUSTRY_HIGH_END_MFG)


class ForecastEngine:
    def __init__(self, store: ObservationStore, config: ForecastConfig) -> None:
        self._store = store
        self._config = config
        self._versions: list[ForecastVersion] = []

    # ---- 版本管理 ----

    @property
    def versions(self) -> tuple[ForecastVersion, ...]:
        return tuple(self._versions)

    @property
    def current(self) -> ForecastVersion | None:
        return self._versions[-1] if self._versions else None

    def get_version(self, version: int) -> ForecastVersion:
        return self._versions[version - 1]

    # ---- 受影响时窗分析 ----

    def affected_windows(
        self, obs: Observation, horizon: tuple[Window, ...]
    ) -> frozenset[tuple[str, Window]]:
        """一条观测（含迟到读数、更正）在预测视野内影响的 (地区, 时窗)。

        依赖关系与 _compute_window 的取数口径一致：
        - 地区总量计量：作为基础负荷，影响视野内同时段（同小时、更晚日期）的窗口；
        - 行业计量：作为增长外推基线，影响不早于该读数的视野窗口；
        - 气象 / 充换电需求：只影响与其重叠的视野窗口；
        - 更正除自身外还覆盖原观测的影响范围。
        """
        affected: set[tuple[str, Window]] = set()
        if obs.kind is ObservationKind.METER and obs.industry is None:
            for window in horizon:
                if (
                    window.start.hour == obs.window.start.hour
                    and window.start.date() > obs.window.start.date()
                ):
                    affected.add((obs.region, window))
        elif obs.kind is ObservationKind.METER:
            for window in horizon:
                if window.start >= obs.window.start:
                    affected.add((obs.region, window))
        elif obs.kind in (ObservationKind.WEATHER, ObservationKind.EV_DEMAND):
            for window in horizon:
                if window.overlaps(obs.window):
                    affected.add((obs.region, window))
        if obs.corrects is not None:
            original = self._store.get(obs.corrects)
            affected |= set(self.affected_windows(original, horizon))
        return frozenset(affected)

    # ---- 计算 ----

    def initial_forecast(
        self,
        regions: tuple[str, ...],
        windows: tuple[Window, ...],
        now: datetime,
        reason: str = "initial",
    ) -> ForecastVersion:
        keys = [(region, window) for region in regions for window in windows]
        return self._build_version(keys, now, reason)

    def recompute(
        self,
        affected: frozenset[tuple[str, Window]] | set[tuple[str, Window]],
        now: datetime,
        reason: str,
    ) -> ForecastVersion:
        """只重算受影响时窗；其余窗口自当前版本结转。"""
        if self.current is None:
            raise ValueError("尚无预测版本，请先执行 initial_forecast")
        return self._build_version(sorted(affected, key=lambda k: (k[0], k[1].start)), now, reason)

    def _build_version(
        self,
        keys_to_compute: list[tuple[str, Window]],
        now: datetime,
        reason: str,
    ) -> ForecastVersion:
        previous = self.current
        results: dict[tuple[str, str], ForecastWindowResult] = (
            dict(previous.results) if previous is not None else {}
        )
        recomputed: list[tuple[str, str]] = []
        for region, window in keys_to_compute:
            result = self._compute_window(region, window)
            results[(region, window.key)] = result
            recomputed.append((region, window.key))
        recomputed_set = set(recomputed)
        carried = tuple(k for k in results if k not in recomputed_set)
        version = ForecastVersion(
            version=(previous.version + 1) if previous is not None else 1,
            created_at=now,
            reason=reason,
            results=results,
            recomputed_keys=tuple(recomputed),
            carried_forward_keys=carried,
        )
        self._versions.append(version)
        return version

    # ---- 单窗口预测 ----

    def _compute_window(self, region: str, window: Window) -> ForecastWindowResult:
        components: dict[str, float] = {}
        inputs: dict[str, tuple[str, ...]] = {}
        assumptions: list[str] = []

        base, base_ids = self._base_load(region, window)
        components["base"] = base
        inputs["base"] = base_ids
        if not base_ids:
            assumptions.append(f"{region} 无同时段历史计量，基础负荷采用兜底值 {base:.1f} MW")

        for industry in TRACKED_INDUSTRIES:
            delta, ind_ids, note = self._industry_growth(region, industry, window)
            components[industry] = delta
            inputs[industry] = ind_ids
            if note:
                assumptions.append(note)

        ev, ev_ids = self._ev_demand(region, window)
        components["ev_charging"] = ev
        inputs["ev_charging"] = ev_ids
        if ev_ids:
            assumptions.append("充换电需求按申报值全额叠加，不参与基线回溯")

        weather, weather_ids, weather_notes = self._weather_adjustment(region, window, base)
        components["weather"] = weather
        inputs["weather"] = weather_ids
        assumptions.extend(weather_notes)

        predicted = sum(components.values())
        return ForecastWindowResult(
            region=region,
            window=window,
            predicted_load_mw=predicted,
            components=components,
            component_inputs=inputs,
            assumptions=tuple(assumptions),
        )

    def _base_load(self, region: str, window: Window) -> tuple[float, tuple[str, ...]]:
        """同一时段历史有效计量的均值；被更正的读数不参与。"""
        history = [
            obs
            for obs in self._store.effective_observations(
                kind=ObservationKind.METER, region=region
            )
            if obs.industry is None
            and "load_mw" in obs.metrics
            and obs.window.start.hour == window.start.hour
            and obs.window.start.date() < window.start.date()
        ]
        if not history:
            fallback = self._config.default_base_mw.get(region, 0.0)
            return fallback, ()
        history.sort(key=lambda o: o.window.start)
        mean = sum(o.metrics["load_mw"] for o in history) / len(history)
        return mean, tuple(o.id for o in history)

    def _industry_growth(
        self, region: str, industry: str, window: Window
    ) -> tuple[float, tuple[str, ...], str | None]:
        """行业增长分量：自基线按日增速外推的增量（不含基线本身，避免与基础负荷重复计）。"""
        growth = self._config.industry_growth_daily.get(industry, 0.0)
        readings = [
            obs
            for obs in self._store.effective_observations(
                kind=ObservationKind.METER, region=region, industry=industry
            )
            if "load_mw" in obs.metrics and obs.window.start <= window.start
        ]
        if readings:
            latest = max(readings, key=lambda o: o.window.start)
            base = latest.metrics["load_mw"]
            base_date = latest.window.start.date()
            ids: tuple[str, ...] = (latest.id,)
        else:
            base = self._config.industry_base_mw.get((region, industry), 0.0)
            base_date = self._config.growth_reference_date
            ids = ()
        if base <= 0.0:
            return 0.0, ids, None
        days = max(0, (window.start.date() - base_date).days)
        delta = base * ((1.0 + growth) ** days - 1.0)
        note = (
            f"{industry} 负荷自 {base_date.isoformat()} 基线 {base:.1f} MW "
            f"按日增速 {growth:.3%} 外推 {days} 天，增量 {delta:.2f} MW"
        )
        return delta, ids, note

    def _ev_demand(self, region: str, window: Window) -> tuple[float, tuple[str, ...]]:
        demands = self._store.effective_observations(
            kind=ObservationKind.EV_DEMAND, region=region, window=window
        )
        total = sum(o.metrics.get("requested_mw", 0.0) for o in demands)
        return total, tuple(o.id for o in demands)

    def _weather_adjustment(
        self, region: str, window: Window, base: float
    ) -> tuple[float, tuple[str, ...], list[str]]:
        notes: list[str] = []
        adjustment = 0.0
        observations = self._store.effective_observations(
            kind=ObservationKind.WEATHER, region=region, window=window
        )
        ids = tuple(o.id for o in observations)
        for obs in observations:
            rain = obs.metrics.get("rain_mm", 0.0)
            temp = obs.metrics.get("temp_c")
            if rain >= self._config.rain_threshold_mm:
                dip = -self._config.rain_dip_ratio * self._config.residential_share * base
                adjustment += dip
                notes.append(
                    f"台风/强降雨（{rain:.1f} mm）导致居民用电短时下降 {abs(dip):.1f} MW，"
                    "按短时扰动处理，不计入长期趋势"
                )
            if temp is not None and temp > self._config.reference_temp_c:
                heat = self._config.heat_sensitivity_mw_per_c * (temp - self._config.reference_temp_c)
                adjustment += heat
                notes.append(
                    f"气温 {temp:.1f}℃ 超过参考值 {self._config.reference_temp_c:.1f}℃，"
                    f"降温负荷增加 {heat:.1f} MW"
                )
        return adjustment, ids, notes
