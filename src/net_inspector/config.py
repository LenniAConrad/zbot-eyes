"""Configuration objects and paths."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"


@dataclass(frozen=True)
class FireClassifierConfig:
    """Configuration for optional fire classifier TFLite model.

    @field input_size Input width/height.
    @field input_channels Number of channels.
    @field positive_index Index for positive class.
    @field score_threshold Minimum score to trigger.
    @field normalize_0_1 Whether to normalize to [0,1].
    """

    input_size: int = 224
    input_channels: int = 3
    positive_index: int = 1
    score_threshold: float = 0.5
    normalize_0_1: bool = True


@dataclass(frozen=True)
class ObjectDetectorConfig:
    """Configuration for optional generic object detector TFLite model.

    @field score_threshold Minimum detection score.
    @field labels_path Optional labels file path.
    """

    score_threshold: float = 0.5
    labels_path: Path = MODELS_DIR / "object_labels.txt"


@dataclass(frozen=True)
class HeuristicConfig:
    """Thresholds for heuristic detection.

    @field green_h_min HSV lower bound for green.
    @field green_h_max HSV upper bound for green.
    @field green_s_min HSV saturation min.
    @field green_v_min HSV value min.
    @field debris_min_area Minimum debris area.
    @field tear_min_area Minimum tear area.
    @field fire_min_area Minimum fire area.
    @field smoke_s_max Max saturation for smoke.
    @field smoke_v_min Min value for smoke.
    @field sky_h_min HSV lower bound for sky.
    @field sky_h_max HSV upper bound for sky.
    @field sky_s_min HSV saturation min for sky.
    @field sky_v_min HSV value min for sky.
    @field sky_min_area_ratio Min sky area ratio.
    @field unknown_min_area_ratio Min unknown area ratio.
    @field tear_green_ratio_max Max green ratio inside tear bbox.
    """

    green_h_min: int = 35
    green_h_max: int = 85
    green_s_min: int = 60
    green_v_min: int = 40

    debris_min_area: int = 120
    tear_min_area: int = 300
    fire_min_area: int = 80
    tear_green_ratio_max: float = 0.22

    smoke_s_max: int = 40
    smoke_v_min: int = 120

    sky_h_min: int = 85
    sky_h_max: int = 135
    sky_s_min: int = 40
    sky_v_min: int = 120

    sky_min_area_ratio: float = 0.08
    unknown_min_area_ratio: float = 0.06


@dataclass(frozen=True)
class AppConfig:
    """Top-level application configuration.

    @field heuristic HeuristicConfig.
    @field fire_classifier FireClassifierConfig.
    @field object_detector ObjectDetectorConfig.
    @field log_path Default JSONL log path.
    @field events_log_path Realtime JSONL log path.
    @field outputs_annotated Annotated output directory.
    @field outputs_raw Raw output directory.
    """

    heuristic: HeuristicConfig = HeuristicConfig()
    fire_classifier: FireClassifierConfig = FireClassifierConfig()
    object_detector: ObjectDetectorConfig = ObjectDetectorConfig()

    log_path: Path = LOG_DIR / "detections.jsonl"
    events_log_path: Path = LOG_DIR / "events.jsonl"
    outputs_annotated: Path = OUTPUT_DIR / "annotated"
    outputs_raw: Path = OUTPUT_DIR / "raw"
