import unittest
from datetime import datetime, timedelta, timezone

from src.models import Observation, ObservationKind, Quality, Window
from src.observations import ObservationStore

UTC = timezone.utc
DAY = datetime(2026, 7, 14, tzinfo=UTC)


def meter(obs_id, hour, load, day=DAY, region="east", corrects=None, quality=Quality.NORMAL):
    window = Window(day.replace(hour=hour, minute=0, second=0, microsecond=0))
    return Observation(
        id=obs_id,
        kind=ObservationKind.METER,
        region=region,
        window=window,
        metrics={"load_mw": load},
        observed_at=window.start,
        ingested_at=window.end,
        corrects=corrects,
        quality=quality,
    )


class ObservationStoreTest(unittest.TestCase):
    def test_append_and_get_raw(self):
        store = ObservationStore()
        obs = meter("m1", 13, 820.0)
        store.append(obs)
        self.assertIs(store.get("m1"), obs)
        self.assertEqual(len(store.all_raw()), 1)

    def test_duplicate_id_rejected(self):
        store = ObservationStore()
        store.append(meter("m1", 13, 820.0))
        with self.assertRaises(ValueError):
            store.append(meter("m1", 14, 830.0))

    def test_correction_preserves_original_and_updates_effective(self):
        store = ObservationStore()
        store.append(meter("m1", 13, 820.0))
        store.append(meter("m2", 13, 800.0, corrects="m1"))
        # 原始观测不改写，仍可取回
        self.assertEqual(store.get("m1").metrics["load_mw"], 820.0)
        self.assertTrue(store.is_superseded("m1"))
        # 有效视图使用更正值
        effective = store.effective_observations(kind=ObservationKind.METER, region="east")
        self.assertEqual([o.id for o in effective], ["m2"])
        self.assertEqual(store.effective("m1").metrics["load_mw"], 800.0)

    def test_correction_chain_resolves_to_latest(self):
        store = ObservationStore()
        store.append(meter("m1", 13, 820.0))
        store.append(meter("m2", 13, 810.0, corrects="m1"))
        store.append(meter("m3", 13, 805.0, corrects="m2"))
        self.assertEqual(store.effective_id("m1"), "m3")

    def test_correction_of_unknown_rejected(self):
        store = ObservationStore()
        with self.assertRaises(ValueError):
            store.append(meter("m9", 13, 800.0, corrects="missing"))

    def test_daily_peak_valley_is_read_only(self):
        store = ObservationStore()
        store.append(meter("m1", 12, 700.0))
        store.append(meter("m2", 13, 820.0))
        store.append(meter("m3", 14, 760.0))
        peak, valley = store.daily_peak_valley("east", DAY.date())
        self.assertEqual(peak[1], 820.0)
        self.assertEqual(valley[1], 700.0)
        # 跨日峰谷计算不改写任何原始观测
        self.assertEqual(store.get("m2").metrics["load_mw"], 820.0)
        self.assertEqual(len(store.all_raw()), 3)

    def test_late_reading_accepted_with_later_ingest(self):
        store = ObservationStore()
        window = Window(DAY.replace(hour=13, minute=0, second=0, microsecond=0))
        late = Observation(
            id="late-1",
            kind=ObservationKind.METER,
            region="east",
            window=window,
            metrics={"load_mw": 815.0},
            observed_at=window.start,
            ingested_at=window.end + timedelta(hours=6),  # 时窗关闭后 6 小时才到
        )
        store.append(late)
        self.assertEqual(store.get("late-1").metrics["load_mw"], 815.0)

    def test_abnormal_observations(self):
        store = ObservationStore()
        store.append(meter("m1", 13, 820.0))
        store.append(meter("m2", 14, 999.0, quality=Quality.ABNORMAL))
        self.assertEqual([o.id for o in store.abnormal_observations()], ["m2"])


if __name__ == "__main__":
    unittest.main()
