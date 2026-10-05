from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from constants import *
from tqdm import tqdm

LOGGER = logging.getLogger(PIPELINE)


@dataclass(frozen=True)
class StageContext:
    input_dir: Path
    output_dir: Path
    run_id: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )

    def stage_dir(self, stage_name: str) -> Path:
        path = self.output_dir / stage_name
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class StageResult:
    stage: str
    success: bool
    output_dir: Path
    metrics_path: Optional[Path] = None
    artifacts: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            STAGE: self.stage,
            SUCCESS: self.success,
            OUTPUT_DIR: str(self.output_dir),
            METRICS_PATH: str(self.metrics_path) if self.metrics_path else None,
            ARTIFACTS: [str(path) for path in self.artifacts],
            WARNINGS: self.warnings,
            ERRORS: self.errors,
        }


class MetricsCollector:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.meta: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, Any] = {}
        self.quality: Dict[str, Optional[float]] = {}
        self.per_file: List[Dict[str, Any]] = []

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + value

    def set_gauge(self, name: str, value: Any) -> None:
        self.gauges[name] = value

    def set_quality(self, name: str, value: Optional[float]) -> None:
        self.quality[name] = value

    def add_file_metrics(self, file_name: str, data: Dict[str, Any]) -> None:
        payload = {"file": file_name, **data}
        self.per_file.append(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            STAGE: self.stage,
            GENERATED_AT: datetime.now(timezone.utc).isoformat(),
            META: self.meta,
            COUNTERS: self.counters,
            GUAGES: self.gauges,
            QUALITY: self.quality,
            PER_FILE: self.per_file,
        }

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class BaseQualityMetric(ABC):
    name: str = "metric"

    @abstractmethod
    def compute(self, raw: str, processed: str) -> Optional[float]:
        raise NotImplementedError


class BaseStage(ABC):
    name: str = "stage"

    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config

    def validate(self, ctx: StageContext) -> None:
        if not ctx.input_dir.exists():
            raise ValueError(f"input_dir does not exist: {ctx.input_dir}")

        if not ctx.input_dir.is_dir():
            raise ValueError(f"input_dir is not a directory: {ctx.input_dir}")

    def list_input_files(self, ctx: StageContext) -> List[Path]:
        return sorted(ctx.input_dir.glob("*.md"))

    @abstractmethod
    def process_file(
        self,
        input_path: Path,
        output_dir: Path,
        metrics: MetricsCollector,
    ) -> Path:
        raise NotImplementedError

    def finalize_metrics(self, metrics: MetricsCollector) -> None:
        pass

    def run(self, ctx: StageContext) -> StageResult:
        output_dir = ctx.stage_dir(self.name)
        metrics = MetricsCollector(self.name)

        if self.config is not None and hasattr(self.config, "to_dict"):
            metrics.set_gauge(CONFIG_NAME, self.config.to_dict())

        result = StageResult(
            stage=self.name,
            success=False,
            output_dir=output_dir,
        )

        try:
            self.validate(ctx)
        except Exception as exc:
            result.errors.append(str(exc))
            metrics_path = output_dir / "metrics.json"
            metrics.save(metrics_path)
            result.metrics_path = metrics_path
            return result

        files = self.list_input_files(ctx)
        metrics.set_gauge(INPUT_FILES, len(files))

        for input_path in tqdm(files, desc=f"[{self.name}] Processing", unit="file", leave=True):
            try:
                artifact = self.process_file(input_path, output_dir, metrics)
                result.artifacts.append(artifact)
                metrics.inc("files_processed")
            except Exception as exc:
                LOGGER.exception("Stage %s failed for %s", self.name, input_path)
                metrics.inc("failed_files")
                result.warnings.append(f"{input_path.name}: {exc}")

        self.finalize_metrics(metrics)

        metrics_path = output_dir / "metrics.json"
        metrics.save(metrics_path)

        result.metrics_path = metrics_path
        result.success = metrics.counters.get(FAILED_FILES, 0) == 0

        return result