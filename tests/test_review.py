import unittest
from datetime import date, datetime, timezone

from src.forecasts import ForecastConfig, ForecastEngine
from src.models import (
    INDUSTRY_DATA_CENTER,
    Alert,
    Enterprise,
    Observation,
    ObservationKind,
    Quality,
    Severity,
    Window,
)
from src.observations import ObservationStore
from src.response import ResponseCoordinator
from src.review import ReviewEngine
from src.risks import AlertEngine, CapacityProfile

UTC = timezone.utc
DAY = datetime(2026, 7, 14, tzinfo=UTC)
H13 = Window(DAY.replace(hour=13))
H14 = Window(DAY.replace(hour=14))
NOW = datetime(2026, 7, 14, 8, 0, tzinfo=UTC)
REVIEW_AT = datetime(2026, 7, 15, 9, 0, tzinfo=UTC)


def make_alert(alert_id, window, margin):
    return Alert(
        id=alert_id,
        region="east",
        window=window,
        severity=Severity.CRITICAL,
        forecast_version=1,
        predicted_load_mw=950.0,
        available_capacity_mw=920.0,
        reserve_margin_mw=margin,
        reserve_margin_ratio=margin / 950.0,
        explanation=(),
        assumptions=(),
        created_at=NOW,
    )


def build_world():
    store = ObservationStore()
    for d in (11, 12, 13):
        day = datetime(2026, 7, d, tzinfo=UTC)
        for hour in (13, 14):
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
    # 充换电申报异常偏高（事后证明是误报来源）
    store.append(
        Observation(
            id="ev-1",
            kind=ObservationKind.EV_DEMAND,
            region="east",
            window=H13,
            metrics={"requested_mw": 60.0},
            observed_at=H13.start,
            ingested_at=H13.start,
            quality=Quality.ABNORMAL,
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
        industry_base_mw={("east", INDUSTRY_DATA_CENTER): 100.0},
        industry_growth_daily={INDUSTRY_DATA_CENTER: 0.003},
        growth_reference_date=date(2026, 7, 1),
    )
    engine = ForecastEngine(store, config)
    version = engine.initial_forecast(("east",), (H13, H14), NOW)

    # 事后更正如期历史读数（v1 预测仍引用原读数）
    window = Window(datetime(2026, 7, 13, 13, tzinfo=UTC))
    store.append(
        Observation(
            id="corr-1",
            kind=ObservationKind.METER,
            region="east",
            window=window,
            metrics={"load_mw": 790.0},
            observed_at=window.start,
            ingested_at=REVIEW_AT,
            corrects="hist-13-13",
        )
    )
    # 实际负荷：H13 远低于预测（误报），H14 高于预测（真实紧张）
    for obs_id, window, load in (("act-13", H13, 850.0), ("act-14", H14, 960.0)):
        store.append(
            Observation(
                id=obs_id,
                kind=ObservationKind.METER,
                region="east",
                window=window,
                metrics={"load_mw": load},
                observed_at=window.start,
                ingested_at=window.end,
            )
        )

    alerts_engine = AlertEngine(store, {"east": CapacityProfile("east", 1000.0)})
    coordinator = ResponseCoordinator(
        (Enterprise("E1", "east", 25.0), Enterprise("E2", "east", 20.0))
    )
    alert13 = make_alert("AL-13", H13, -25.0)
    alert14 = make_alert("AL-14", H14, -30.0)
    i13 = coordinator.plan(alert13, NOW)
    i14 = coordinator.plan(alert14, NOW)
    # H13：E1 足额履约
    coordinator.record_outcome(i13[0].id, 25.0, REVIEW_AT)
    # H14：E2 退出响应；E1 部分履约；另有一笔指令被调度撤回
    coordinator.enterprise_exit(i14[1].id, NOW, reason="产线无法停机")
    extra = coordinator.plan(alert14, NOW)  # 退出后缺口仍在，重新补发
    coordinator.dispatch_withdraw(extra[0].id, NOW, reason="指令重复")
    coordinator.record_outcome(i14[0].id, 20.0, REVIEW_AT)

    review = ReviewEngine(store, alerts_engine)
    return version, (alert13, alert14), coordinator, review


class ReviewEngineTest(unittest.TestCase):
    def test_review_totals_and_rows(self):
        version, alerts, coordinator, review = build_world()
        report = review.review((("east", H13), ("east", H14)), version, alerts, coordinator, REVIEW_AT)

        self.assertEqual(report.alert_count, 2)
        self.assertEqual(report.false_alarm_count, 1)
        # 承诺：H13 E1 25 + H14 E1 25 + H14 E2 退出 5（撤回的不计入）
        self.assertAlmostEqual(report.total_committed_mw, 55.0)
        self.assertAlmostEqual(report.total_actual_shed_mw, 45.0)
        self.assertAlmostEqual(report.total_unfulfilled_mw, 10.0)

        by_window = {r.window: r for r in report.rows}
        h13 = by_window[H13]
        self.assertTrue(h13.alert_issued)
        self.assertTrue(h13.false_alarm)
        self.assertTrue(h13.actual_reserve_ok)
        self.assertAlmostEqual(h13.actual_mw, 850.0)
        self.assertIsNotNone(h13.forecast_error_mw)

        h14 = by_window[H14]
        self.assertTrue(h14.alert_issued)
        self.assertFalse(h14.false_alarm)
        self.assertAlmostEqual(h14.unfulfilled_mw, 10.0)

    def test_false_alarm_attributed_to_abnormal_and_corrected_data(self):
        version, alerts, coordinator, review = build_world()
        report = review.review((("east", H13),), version, alerts, coordinator, REVIEW_AT)
        row = report.rows[0]
        causes = "；".join(row.false_alarm_causes)
        self.assertIn("ev-1", causes)          # 异常充换电申报
        self.assertIn("abnormal", causes)
        self.assertIn("hist-13-13", causes)    # 事后被更正的历史读数
        self.assertIn("corr-1", causes)

    def test_withdrawn_instruction_not_counted_as_commitment(self):
        version, alerts, coordinator, review = build_world()
        report = review.review((("east", H14),), version, alerts, coordinator, REVIEW_AT)
        row = report.rows[0]
        # 撤回的 5 MW 不计入承诺；退出与部分履约构成未履约
        self.assertAlmostEqual(row.committed_mw, 30.0)
        self.assertAlmostEqual(row.actual_shed_mw, 20.0)
        self.assertAlmostEqual(row.unfulfilled_mw, 10.0)


if __name__ == "__main__":
    unittest.main()
