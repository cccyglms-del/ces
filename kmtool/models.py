from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class AxisBounds:
    x_min: float = 0.0
    x_max: float = 100.0
    y_min: float = 0.0
    y_max: float = 1.0
    time_unit: str = "months"


@dataclass
class CurveExtractionRequest:
    source_type: str
    file_name: str
    study_id: str
    panel_hint: Optional[str] = None
    time_unit: Optional[str] = None
    manual_crop: Optional[Tuple[int, int, int, int]] = None
    manual_axis_bounds: Optional[AxisBounds] = None


@dataclass
class ArmMapping:
    curve_id: str
    arm_label: str
    detected_color: Tuple[int, int, int]
    source: str = "cv"


@dataclass
class RiskTableRow:
    time: float
    arm_counts: Dict[str, int]


@dataclass
class CurveSeries:
    curve_id: str
    arm_label: str
    detected_color: Tuple[int, int, int]
    pixel_points: List[Tuple[int, int]]
    data_points: List[Tuple[float, float]]
    confidence: float
    warnings: List[str] = field(default_factory=list)
    detected_censor_count: int = 0


@dataclass
class ExtractionReview:
    arm_mappings: List[ArmMapping]
    axis_bounds: AxisBounds
    risk_table_rows: List[RiskTableRow]
    confidence_score: float
    warnings: List[str] = field(default_factory=list)
    ocr_text: str = ""
    llm_notes: List[str] = field(default_factory=list)


@dataclass
class ReconstructedArmData:
    study_id: str
    comparison_id: str
    arm_label: str
    time: List[float]
    event: List[int]
    source_curve_id: str
    reconstruction_method: str
    confidence: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class PairwiseResult:
    comparison_id: str
    log_rank_p: float
    hr: float
    hr_ci_low: float
    hr_ci_high: float
    n_reconstructed: int
    warnings: List[str] = field(default_factory=list)


@dataclass
class StudyCandidate:
    source: str
    study_id: str
    title: str
    abstract: str
    year: Optional[int] = None
    journal: str = ""
    pmid: str = ""
    doi: str = ""
    comparison_type: str = ""
    treatments: List[str] = field(default_factory=list)
    endpoint_text: str = ""
    population_text: str = ""
    open_access_url: str = ""
    reported_hr: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class StudyEffect:
    study_id: str
    comparison_label: str
    treatment_left: str
    treatment_right: str
    log_hr: float
    se: float
    source_method: str
    endpoint_text: str = ""
    population_text: str = ""
    warnings: List[str] = field(default_factory=list)


@dataclass
class IndirectComparisonRequest:
    treatment_a: str
    treatment_b: str
    treatment_c: str
    endpoint: str
    selected_ab_studies: List[str]
    selected_bc_studies: List[str]
    population_hint: str = ""
    allow_inconsistent: bool = False


@dataclass
class IndirectComparisonResult:
    ac_log_hr: float
    ac_hr: float
    ci95: Tuple[float, float]
    study_provenance: List[str]
    heterogeneity_notes: List[str]
    warnings: List[str] = field(default_factory=list)


def dataclass_to_dict(instance):
    return asdict(instance)
