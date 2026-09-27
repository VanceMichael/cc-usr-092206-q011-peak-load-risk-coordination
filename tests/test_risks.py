import unittest
from datetime import date, datetime, timezone

from src.forecasts import ForecastConfig, ForecastEngine
from src.models import (
    INDUSTRY_DATA_CENTER,
    Observation,
    ObservationKind,
    Severity,
    Window,
)
from src.observations import ObservationStore
from src.risks import AlertEngine, AlertThresholds, CapacityProfile, explain_alert

UTC = timezone.utc
HORIZON_DAY = datetime(2026, 7, 14, tzinfo=UTC)
H12 = Window(HORIZON_DAY.replace(hour=12))
H13 = Window(HORIZON_DAY.replace(hour=13))
H14 = Window(HORIZON_DAY.replace(hour=14))
HORIZON = (H12, H13, H14)
NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)


def build():
    store = ObservationStore()
    for d in (11, 12, 13):
        day = datetime(2026, 7, d, tzinfo=UTC)
        for hour in (12, 13, 14):
            window = Window(day.replace(hour=hour))
            store.append(
                Observation(
                    id=f"hist-{d}-{hour}",
                    kind=ObservationKind.METER,
                    region="east",
                    window=window,
                    metrics={"load_mw": 820.0},
                    observed_at=window.start,
                    ingested_at=window.end,
                )
            )
    store.append(
        Observation(
            id="ev-1",
            kind=ObservationKind.EV_DEMAND,
            region="east",
            window=H13,
            metrics={"requested_mw": 60.0},
            observed_at=H13.start,
            ingested_at=H13.start,
        )
    )
    store.append(
        Observation(
            id="wx-heat",
            kind=ObservationKind.WEATHER,
            region="east",
            window=H14,
            metrics={"temp_c": 38.0, "rain_mm": 0.0},
            observed_at=H14.start,
            ingested_at=H14.start,
        )
    )
    store.append(
        Observation(
            id="mnt-1",
            kind=ObservationKind.MAINTENANCE,
            region="east",
            window=H14,
            metrics={"outage_mw": 80.0},
            observed_at=H14.start,
            ingested_at=NOW,
        )
    )
    config = ForecastConfig(
        default_base_mw={"east": 800.0},
        industry_base_mw={
            ("east", INDUSTRY_DATA_CENTER): 100.0,
            ("east", "high_end_manufacturing"): 150.0,
        },
        industry_growth_daily={INDUSTRY_DATA_CENTER: 0.003, "high_end_manufacturing": 0.001},
        growth_reference_date=date(2026, 7, 1),
    )
    engine = ForecastEngine(store, config)
    version = engine.initial_forecast(("east",), HORIZON, NOW)
    alerts_engine = AlertEngine(store, {"east": CapacityProfile("east", 1000.0)})
    return version, alerts_engine


class AlertEngineTest(unittest.TestCase):
    def test_alerts_and_severity(self):
        version, engine = build()
        alerts = engine.evaluate(version, NOW)
        by_window = {a.window: a for a in alerts}
        self.assertEqual(set(by_window), {H13, H14})
        # H13 裕度比约 12.9%（提示档），H14 检修叠加高温为负裕度（严重档）
        self.assertEqual(by_window[H13].severity, Severity.ADVISORY)
        self.assertEqual(by_window[H14].severity, Severity.CRITICAL)
        # 检修停运后可用容量 = 1000 - 80
        self.assertAlmostEqual(by_window[H14].available_capacity_mw, 920.0)
        self.assertLess(by_window[H14].reserve_margin_mw, 0.0)

    def test_explanation_is_itemized_with_sources(self):
        version, engine = build()
        alerts = engine.evaluate(version, NOW)
        alert = next(a for a in alerts if a.window == H14)
        labels = [item.label for item in alert.explanation]
        self.assertIn("预测负荷", labels)
        self.assertIn("分量·基础负荷", labels)
        self.assertIn("分量·数据中心", labels)
        self.assertIn("分量·高端制造", labels)
        self.assertIn("分量·气象修正", labels)
        self.assertIn("可用容量", labels)
        self.assertIn("备用裕度", labels)
        capacity_item = next(i for i in alert.explanation if i.label == "可用容量")
        self.assertIn("mnt-1", capacity_item.source_observation_ids)
        weather_item = next(i for i in alert.explanation if i.label == "分量·气象修正")
        self.assertIn("wx-heat", weather_item.source_observation_ids)
        self.assertIn("检修计划按申报停运容量全额扣减", alert.assumptions)

    def test_explain_alert_dict(self):
        version, engine = build()
        alerts = engine.evaluate(version, NOW)
        doc = explain_alert(alerts[0])
        self.assertEqual(doc["alert_id"], alerts[0].id)
        self.assertTrue(doc["items"])
        self.assertIn("assumptions", doc)
        for item in doc["items"]:
            self.assertIn("label", item)
            self.assertIn("value", item)
            self.assertIn("sources", item)

    def test_no_alert_when_margin_healthy(self):
        version, _ = build()
        store = ObservationStore()
        engine = AlertEngine(store, {"east": CapacityProfile("east", 5000.0)})
        self.assertEqual(engine.evaluate(version, NOW), ())

    def test_severity_thresholds(self):
        engine = AlertEngine(
            ObservationStore(), {}, AlertThresholds(0.15, 0.10, 0.05)
        )
        self.assertIsNone(engine.severity_for(0.15))
        self.assertEqual(engine.severity_for(0.149), Severity.ADVISORY)
        self.assertEqual(engine.severity_for(0.099), Severity.WARNING)
        self.assertEqual(engine.severity_for(0.049), Severity.CRITICAL)


if __name__ == "__main__":
    unittest.main()
