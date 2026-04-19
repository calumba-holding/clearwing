"""Severity calibration tracking (spec 009).

Tracks agreement between discoverer, validator, and human severity
assessments over time. Target: 89% exact match (Glasswing reference).
"""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


@dataclass(frozen=True)
class CalibrationRecord:
    finding_id: str
    session_id: str
    cwe: str
    discoverer_severity: str
    validator_severity: str | None = None
    human_severity: str | None = None
    axes: dict[str, bool] = field(default_factory=dict)
    timestamp: str = ""
    exact_match: bool | None = None
    within_one: bool | None = None


class CalibrationStore:
    """Append-only JSONL store for severity calibration records."""

    @staticmethod
    def _default_path() -> Path:
        from clearwing.core.config import clearwing_home

        return clearwing_home() / "sourcehunt" / "calibration.jsonl"

    def __init__(self, path: Path | str | None = None):
        self._path = Path(path) if path else self._default_path()

    def append(self, record: CalibrationRecord) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), default=str) + "\n"
        # Atomic append: write to temp file, then append
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            logger.warning("Failed to write calibration record", exc_info=True)

    def record_human_verdict(
        self,
        finding_id: str,
        session_id: str,
        human_severity: str,
    ) -> None:
        records = self.load_all()
        updated = []
        for r in records:
            if r.finding_id == finding_id and r.session_id == session_id:
                d = asdict(r)
                d["human_severity"] = human_severity
                # Compute match metrics
                if r.validator_severity:
                    d["exact_match"] = r.validator_severity == human_severity
                    d_rank = _SEVERITY_RANK.get(r.validator_severity, 0)
                    h_rank = _SEVERITY_RANK.get(human_severity, 0)
                    d["within_one"] = abs(d_rank - h_rank) <= 1
                updated.append(CalibrationRecord(**d))
            else:
                updated.append(r)
        self._write_all(updated)

    def load_all(self) -> list[CalibrationRecord]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                records.append(CalibrationRecord(**{
                    k: v for k, v in data.items()
                    if k in CalibrationRecord.__dataclass_fields__
                }))
            except (json.JSONDecodeError, TypeError):
                continue
        return records

    def stats(self) -> dict[str, Any]:
        records = self.load_all()
        with_human = [r for r in records if r.human_severity is not None]
        if not with_human:
            return {
                "total_records": len(records),
                "human_reviewed": 0,
                "exact_match_rate": 0.0,
                "within_one_rate": 0.0,
            }
        exact = sum(1 for r in with_human if r.exact_match is True)
        within = sum(1 for r in with_human if r.within_one is True)
        return {
            "total_records": len(records),
            "human_reviewed": len(with_human),
            "exact_match_rate": exact / len(with_human),
            "within_one_rate": within / len(with_human),
        }

    def _write_all(self, records: list[CalibrationRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(asdict(r), default=str) + "\n")
        tmp.replace(self._path)
