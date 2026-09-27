"""高峰用电风险协调平台的核心数据模型。

约定：
- 所有时间均为带时区的 datetime（测试与示例统一使用 UTC）。
- 观测（Observation）一旦写入即不可改写；计量更正以新的观测链接原观测表示。
- 预测按版本保存；每个时窗结果记录逐项输入与假设，供预警解释与复盘使用。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


# 行业标识常量（充换电需求以独立观测类型接入，不在此列）
INDUSTRY_DATA_CENTER = "data_center"            # 数据中心 / 互联网数据服务
INDUSTRY_HIGH_END_MFG = "high_end_manufacturing"  # 高端制造


class ObservationKind(str, Enum):
    METER = "meter"              # 地区/行业计量
    WEATHER = "weather"          # 气象观测
    MAINTENANCE = "maintenance"  # 设备检修
    EV_DEMAND = "ev_demand"      # 充换电需求


class Quality(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"
    ABNORMAL = "abnormal"


@dataclass(frozen=True)
class Window:
    """预测与调度的最小时窗，默认 1 小时。"""

    start: datetime
    hours: int = 1

    @property
    def end(self) -> datetime:
        return self.start + timedelta(hours=self.hours)

    @property
    def key(self) -> str:
        return f"{self.start.isoformat()}/{self.hours}h"

    def contains(self, ts: datetime) -> bool:
        return self.start <= ts < self.end

    def overlaps(self, other: "Window") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass(frozen=True)
class Observation:
    """一条原始观测。写入后不可改写；更正通过 corrects 链接原观测。"""

    id: str
    kind: ObservationKind
    region: str
    window: Window
    metrics: dict[str, float]           # 如 {"load_mw": 812.5} / {"rain_mm": 34.0, "temp_c": 26.5}
    observed_at: datetime               # 读数所属时刻
    ingested_at: datetime               # 平台接收时刻（迟到读数晚于时窗关闭）
    industry: str | None = None
    source: str = ""
    corrects: str | None = None         # 被更正的原观测 id
    quality: Quality = Quality.NORMAL
    note: str = ""


@dataclass(frozen=True)
class ForecastWindowResult:
    """单个时窗的预测结果，含逐项分量、输入观测与假设。"""

    region: str
    window: Window
    predicted_load_mw: float
    components: dict[str, float]                  # base / 各行业 / ev_charging / weather
    component_inputs: dict[str, tuple[str, ...]]  # 每个分量用到的观测 id
    assumptions: tuple[str, ...]

    @property
    def input_observation_ids(self) -> tuple[str, ...]:
        seen: list[str] = []
        for ids in self.component_inputs.values():
            for oid in ids:
                if oid not in seen:
                    seen.append(oid)
        return tuple(seen)


@dataclass(frozen=True)
class ForecastVersion:
    """一次完整的预测快照。重算只替换受影响时窗，其余窗口自上一版本结转。"""

    version: int
    created_at: datetime
    reason: str                                    # initial / late-reading / correction / new-forecast
    results: dict[tuple[str, str], ForecastWindowResult]  # (region, window.key) -> 结果
    recomputed_keys: tuple[tuple[str, str], ...]
    carried_forward_keys: tuple[tuple[str, str], ...]

    def result_for(self, region: str, window: Window) -> ForecastWindowResult | None:
        return self.results.get((region, window.key))


class Severity(str, Enum):
    ADVISORY = "advisory"  # 提示
    WARNING = "warning"    # 预警
    CRITICAL = "critical"  # 严重


@dataclass(frozen=True)
class ExplanationItem:
    """预警解释中的一项：数据或假设 + 其来源观测。"""

    label: str
    value: str
    source_observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Alert:
    """日内预警。解释逐项列出所用数据与假设，供分析人员核对。"""

    id: str
    region: str
    window: Window
    severity: Severity
    forecast_version: int
    predicted_load_mw: float
    available_capacity_mw: float
    reserve_margin_mw: float
    reserve_margin_ratio: float
    explanation: tuple[ExplanationItem, ...]
    assumptions: tuple[str, ...]
    created_at: datetime


class InstructionStatus(str, Enum):
    ISSUED = "issued"              # 已签发
    ACKNOWLEDGED = "acknowledged"  # 企业已确认
    FULFILLED = "fulfilled"        # 已履约
    PARTIAL = "partial"            # 部分履约
    EXITED = "exited"              # 企业退出响应
    WITHDRAWN = "withdrawn"        # 调度撤回指令


TERMINAL_STATUSES = frozenset(
    {
        InstructionStatus.FULFILLED,
        InstructionStatus.PARTIAL,
        InstructionStatus.EXITED,
        InstructionStatus.WITHDRAWN,
    }
)


@dataclass(frozen=True)
class Instruction:
    """需求响应指令。状态迁移产生新实例，历史事件保留在 history 中。"""

    id: str
    enterprise_id: str
    region: str
    window: Window
    requested_reduction_mw: float
    status: InstructionStatus
    issued_at: datetime
    alert_id: str | None = None
    actual_shed_mw: float | None = None
    history: tuple[tuple[datetime, str], ...] = ()


@dataclass(frozen=True)
class Enterprise:
    id: str
    region: str
    max_reduction_mw: float
    industries: tuple[str, ...] = ()


@dataclass(frozen=True)
class WindowReview:
    """单个时窗的事后复盘。"""

    region: str
    window: Window
    forecast_mw: float | None
    actual_mw: float | None
    forecast_error_mw: float | None
    alert_issued: bool
    actual_reserve_ok: bool | None
    false_alarm: bool
    false_alarm_causes: tuple[str, ...]
    committed_mw: float
    actual_shed_mw: float
    unfulfilled_mw: float


@dataclass(frozen=True)
class ReviewReport:
    generated_at: datetime
    rows: tuple[WindowReview, ...]
    total_committed_mw: float
    total_actual_shed_mw: float
    total_unfulfilled_mw: float
    alert_count: int
    false_alarm_count: int
