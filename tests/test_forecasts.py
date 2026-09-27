import unittest
from datetime import date, datetime, timezone

from src.forecasts import ForecastConfig, ForecastEngine
from src.models import (
    INDUSTRY_DATA_CENTER,
    Observation,
    ObservationKind,
    Quality,
    Window,
)
from src.observations import ObservationStore

UTC = timezone.utc
HORIZON_DAY = datetime(2026, 7, 14, tzinfo=UTC)
H12 = Window(HORIZON_DAY.replace(hour=12))
H13 = Window(HORIZON_DAY.replace(hour=13))
H14 = Window(HORIZON_DAY.replace(hour=14))
HORIZON = (H12, H13, H14)
NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def make_config():
    return ForecastConfig(
        default_base_mw={"east": 800.0},
        industry_base_mw={
            ("east", INDUSTRY_DATA_CENTER): 100.0,
            ("east", "high_end_manufacturing"): 150.0,
        },
        industry_growth_daily={INDUSTRY_DATA_CENTER: 0.003, "high_end_manufacturing": 0.001},
        growth_reference_date=date(2026, 7, 1),
    )


def meter(obs_id, day, hour, load, corrects=None, quality=Quality.NORMAL):
    window = Window(day.replace(hour=hour, minute=0, second=0, microsecond=0))
    return Observation(
        id=obs_id,
        kind=ObservationKind.METER,
        region="east",
        window=window,
        metrics={"load_mw": load},
        observed_at=window.start,
        ingested_at=window.end,
        corrects=corrects,
        quality=quality,
    )


def weather(obs_id, window, **metrics):
    return Observation(
        id=obs_id,
        kind=ObservationKind.WEATHER,
        region="east",
        window=window,
        metrics=metrics,
        observed_at=window.start,
        ingested_at=window.start,
    )


def ev_demand(obs_id, window, requested_mw):
    return Observation(
        id=obs_id,
        kind=ObservationKind.EV_DEMAND,
        region="east",
        window=window,
        metrics={"requested_mw": requested_mw},
        observed_at=window.start,
        ingested_at=window.start,
    )


def seeded_engine():
    store = ObservationStore()
    for d in (11, 12, 13):
        day = datetime(2026, 7, d, tzinfo=UTC)
        for hour in (12, 13, 14):
            store.append(meter(f"hist-{d}-{hour}", day, hour, 820.0))
    store.append(weather("wx-rain", H12, rain_mm=25.0, temp_c=26.0))
    store.append(ev_demand("ev-1", H13, 60.0))
    store.append(weather("wx-heat", H14, rain_mm=0.0, temp_c=38.0))
    return ForecastEngine(store, make_config()), store


class ForecastEngineTest(unittest.TestCase):
    def test_initial_forecast_components(self):
        engine, _ = seeded_engine()
        v1 = engine.initial_forecast(("east",), HORIZON, NOW)
        self.assertEqual(v1.version, 1)

        h12 = v1.result_for("east", H12)
        self.assertAlmostEqual(h12.components["base"], 820.0)
        self.assertAlmostEqual(h12.components["weather"], -49.2)  # 0.2*0.3*820
        self.assertAlmostEqual(h12.components["ev_charging"], 0.0)
        self.assertAlmostEqual(h12.predicted_load_mw, 820.0 + 3.9709 + 1.9617 - 49.2, places=3)

        h13 = v1.result_for("east", H13)
        self.assertAlmostEqual(h13.components["ev_charging"], 60.0)
        self.assertAlmostEqual(h13.predicted_load_mw, 820.0 + 3.9709 + 1.9617 + 60.0, places=3)

        h14 = v1.result_for("east", H14)
        self.assertAlmostEqual(h14.components["weather"], 125.0)  # 25 MW/℃ * (38-33)
        self.assertAlmostEqual(h14.predicted_load_mw, 820.0 + 3.9709 + 1.9617 + 125.0, places=3)

    def test_typhoon_rain_dip_is_transient_assumption(self):
        engine, _ = seeded_engine()
        v1 = engine.initial_forecast(("east",), HORIZON, NOW)
        h12 = v1.result_for("east", H12)
        notes = "；".join(h12.assumptions)
        self.assertIn("短时扰动", notes)
        self.assertIn("不计入长期趋势", notes)
        # 气象修正只落在降雨时窗，其他窗口不受影响
        self.assertAlmostEqual(v1.result_for("east", H13).components["weather"], 0.0)

    def test_industry_growth_assumption_recorded(self):
        engine, _ = seeded_engine()
        v1 = engine.initial_forecast(("east",), HORIZON, NOW)
        h13 = v1.result_for("east", H13)
        self.assertTrue(any("data_center" in a and "日增速" in a for a in h13.assumptions))
        self.assertGreater(h13.components[INDUSTRY_DATA_CENTER], 0.0)

    def test_affected_windows_mapping(self):
        engine, _ = seeded_engine()
        # 历史同时段读数 -> 视野内同小时的未来窗口
        late = meter("late-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 840.0)
        self.assertEqual(engine.affected_windows(late, HORIZON), frozenset({("east", H14)}))
        # 气象只影响重叠窗口
        wx = weather("wx-2", H13, rain_mm=30.0)
        self.assertEqual(engine.affected_windows(wx, HORIZON), frozenset({("east", H13)}))
        # 行业计量 -> 不早于读数的全部视野窗口
        ind = Observation(
            id="ind-1",
            kind=ObservationKind.METER,
            region="east",
            industry=INDUSTRY_DATA_CENTER,
            window=Window(datetime(2026, 7, 13, 10, tzinfo=UTC)),
            metrics={"load_mw": 105.0},
            observed_at=datetime(2026, 7, 13, 10, tzinfo=UTC),
            ingested_at=datetime(2026, 7, 13, 11, tzinfo=UTC),
        )
        self.assertEqual(
            engine.affected_windows(ind, HORIZON),
            frozenset({("east", H12), ("east", H13), ("east", H14)}),
        )

    def test_recompute_only_affected_window(self):
        engine, store = seeded_engine()
        v1 = engine.initial_forecast(("east",), HORIZON, NOW)
        late = meter("late-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 840.0)
        store.append(late)
        affected = engine.affected_windows(late, HORIZON)
        v2 = engine.recompute(affected, NOW, "late-reading")

        self.assertEqual(v2.version, 2)
        self.assertEqual(set(v2.recomputed_keys), {("east", H14.key)})
        self.assertEqual(
            set(v2.carried_forward_keys), {("east", H12.key), ("east", H13.key)}
        )
        # 未受影响窗口原样结转（同一对象）
        self.assertIs(v2.result_for("east", H12), v1.result_for("east", H12))
        self.assertIs(v2.result_for("east", H13), v1.result_for("east", H13))
        # 受影响窗口按新基线重算：(820*3+840)/4 = 825
        self.assertAlmostEqual(v2.result_for("east", H14).components["base"], 825.0)

    def test_correction_recomputes_window_and_keeps_original(self):
        engine, store = seeded_engine()
        v1 = engine.initial_forecast(("east",), HORIZON, NOW)
        late = meter("late-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 840.0)
        store.append(late)
        engine.recompute(engine.affected_windows(late, HORIZON), NOW, "late-reading")

        correction = meter("corr-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 800.0, corrects="late-1")
        store.append(correction)
        v3 = engine.recompute(engine.affected_windows(correction, HORIZON), NOW, "correction")

        self.assertEqual(v3.reason, "correction")
        # (820*3+800)/4 = 815
        self.assertAlmostEqual(v3.result_for("east", H14).components["base"], 815.0)
        # 原始迟到读数未被改写
        self.assertEqual(store.get("late-1").metrics["load_mw"], 840.0)
        self.assertEqual(store.effective("late-1").metrics["load_mw"], 800.0)

    def test_recompute_before_initial_rejected(self):
        engine, _ = seeded_engine()
        with self.assertRaises(ValueError):
            engine.recompute(frozenset({("east", H14)}), NOW, "late-reading")

    def test_versions_are_kept(self):
        engine, store = seeded_engine()
        engine.initial_forecast(("east",), HORIZON, NOW)
        late = meter("late-1", datetime(2026, 7, 10, tzinfo=UTC), 14, 840.0)
        store.append(late)
        engine.recompute(engine.affected_windows(late, HORIZON), NOW, "late-reading")
        self.assertEqual(len(engine.versions), 2)
        self.assertEqual(engine.get_version(1).version, 1)
        self.assertEqual(engine.current.version, 2)


if __name__ == "__main__":
    unittest.main()
