"""
EEG TXT Parser
==============
Heuristically parses poorly-structured EEG text files with:
  - Unknown / mixed delimiters
  - Inconsistent column counts
  - Corrupted or incomplete rows
  - Optional / malformed headers
  - Missing or irregular timestamps

Usage
-----
>>> from biosignal_framework.ingestion.eeg_txt_parser import EEGTxtParser
>>> parser = EEGTxtParser("eeg_data.txt")
>>> df, stats = parser.parse()
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Configuration

@dataclass
class EEGParserConfig:
    """Configurable knobs for the EEG TXT parser."""

    delimiter_override: Optional[str] = None
    expected_channels: Optional[int] = None
    channel_names: Optional[List[str]] = None
    timestamp_col: Optional[int] = None          # 0-based index
    has_header: Optional[bool] = None
    fill_strategy: str = "interpolate"
    max_bad_row_fraction: float = 0.20
    encoding: str = "utf-8"
    sampling_rate: Optional[float] = None

# Parse statistics container

@dataclass
class ParseStats:
    """Parsing diagnostics returned alongside the DataFrame."""

    total_rows_read: int = 0
    valid_rows: int = 0
    rejected_rows: int = 0
    delimiter_detected: str = ""
    header_detected: bool = False
    sampling_rate_inferred: float = 0.0
    missing_values_filled: int = 0
    channel_names: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def recovery_rate(self) -> float:
        """Fraction of rows successfully recovered."""
        if self.total_rows_read == 0:
            return 0.0
        return self.valid_rows / self.total_rows_read

# Main parser class

class EEGTxtParser:
    """Robust, heuristic parser for badly-formatted EEG text files."""

    _DELIMITER_CANDIDATES: Tuple[str, ...] = ("\t", ",", ";", " ", "|")
    _NUMERIC_RE = re.compile(r"^[+-]?\d+(\.\d+)?([eE][+-]?\d+)?$")
    _CLEAN_RE = re.compile(r"[^\x20-\x7E\t\n\r]")

    def __init__(
        self,
        filepath: str | Path,
        config: Optional[EEGParserConfig] = None,
    ) -> None:
        self.filepath = Path(filepath)
        self.config = config or EEGParserConfig()
        self._stats = ParseStats()

    # Public API

    def parse(self) -> Tuple[pd.DataFrame, ParseStats]:
        """Parse the EEG TXT file and return a clean DataFrame + statistics."""
        logger.info("Parsing EEG file: %s", self.filepath)
        raw_lines = self._read_raw_lines()
        delimiter = self._detect_delimiter(raw_lines)
        self._stats.delimiter_detected = repr(delimiter)
        logger.debug("Delimiter detected: %s", repr(delimiter))

        has_header, header_line_idx = self._detect_header(raw_lines, delimiter)
        self._stats.header_detected = has_header

        data_lines = raw_lines[header_line_idx + 1:] if has_header else raw_lines[header_line_idx:]

        n_data_cols = self._infer_col_count_from_data(data_lines, delimiter)
        column_names = self._extract_column_names(
            raw_lines, header_line_idx, has_header, delimiter, n_data_cols
        )

        records, rejected = self._parse_rows(data_lines, delimiter, len(column_names))
        self._stats.total_rows_read = len(data_lines)
        self._stats.valid_rows = len(records)
        self._stats.rejected_rows = rejected

        bad_frac = rejected / max(len(data_lines), 1)
        if bad_frac > self.config.max_bad_row_fraction:
            warn = (
                f"{rejected}/{len(data_lines)} rows rejected "
                f"({bad_frac:.1%}) — exceeds threshold "
                f"{self.config.max_bad_row_fraction:.1%}"
            )
            logger.warning(warn)
            self._stats.warnings.append(warn)

        df = pd.DataFrame(records, columns=column_names)
        df = self._apply_column_names(df)
        df, ts_col = self._extract_timestamps(df)
        df = self._coerce_numerics(df)

        if ts_col is not None:
            df.index = ts_col
            df.index.name = "timestamp_s"

        df, filled = self._handle_missing(df)
        self._stats.missing_values_filled = filled

        sr = self._infer_sampling_rate(df.index)
        self._stats.sampling_rate_inferred = sr
        self._stats.channel_names = list(df.columns)

        logger.info(
            "EEG parse complete — %d valid rows, %d rejected, SR=%.1f Hz, %d values filled",
            self._stats.valid_rows, self._stats.rejected_rows, sr, filled,
        )
        return df, self._stats

    # Internal helpers

    def _read_raw_lines(self) -> List[str]:
        try:
            text = self.filepath.read_text(encoding=self.config.encoding, errors="replace")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"EEG file not found: {self.filepath}") from exc

        text = self._CLEAN_RE.sub("", text)
        lines = [ln.rstrip("\r\n") for ln in text.splitlines()]
        lines = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        return lines

    def _score_delimiter(self, line: str, delim: str) -> int:
        parts = line.split(delim)
        if len(parts) < 2:
            return 0
        numeric_count = sum(1 for p in parts if self._NUMERIC_RE.match(p.strip()))
        return numeric_count

    def _detect_delimiter(self, lines: List[str]) -> str:
        if self.config.delimiter_override:
            return self.config.delimiter_override

        sample = [ln for ln in lines if ln.strip()][:20]
        scores: Dict[str, int] = {d: 0 for d in self._DELIMITER_CANDIDATES}
        for line in sample:
            for delim in self._DELIMITER_CANDIDATES:
                scores[delim] += self._score_delimiter(line, delim)

        best = max(scores, key=lambda d: scores[d])
        if scores[best] == 0:
            logger.warning("No delimiter scored >0; falling back to whitespace.")
            return r"\s+"
        return best

    def _detect_header(self, lines: List[str], delimiter: str) -> Tuple[bool, int]:
        if self.config.has_header is not None:
            return self.config.has_header, 0

        first_line = lines[0] if lines else ""
        parts = re.split(re.escape(delimiter) if delimiter != r"\s+" else r"\s+", first_line.strip())
        numeric_parts = sum(1 for p in parts if self._NUMERIC_RE.match(p.strip()))
        has_header = numeric_parts < len(parts) * 0.5
        return has_header, 0

    def _extract_column_names(
        self,
        lines: List[str],
        header_idx: int,
        has_header: bool,
        delimiter: str,
        n_data_cols: int = 0,
    ) -> List[str]:
        if self.config.channel_names:
            return self.config.channel_names

        if has_header:
            header_line = lines[header_idx]
            sep = re.escape(delimiter) if delimiter != r"\s+" else r"\s+"
            names = [c.strip().strip('"\'') for c in re.split(sep, header_line)]
            names = [n if n else f"col_{i}" for i, n in enumerate(names)]

            if n_data_cols > 1 and len(names) < n_data_cols:
                ws_names = [c.strip().strip('"\'') for c in header_line.split()]
                if len(ws_names) >= n_data_cols:
                    names = ws_names

            if n_data_cols and len(names) > n_data_cols:
                names = names[:n_data_cols]
            elif n_data_cols and len(names) < n_data_cols:
                names += [f"col_{i}" for i in range(len(names), n_data_cols)]

            return names

        return [f"col_{i}" for i in range(n_data_cols or 1)]

    def _infer_col_count_from_data(self, data_lines: List[str], delimiter: str) -> int:
        sep = re.escape(delimiter) if delimiter != r"\s+" else r"\s+"
        counts: List[int] = []
        for line in data_lines[:20]:
            parts = [p for p in re.split(sep, line.strip()) if p.strip()]
            if parts:
                counts.append(len(parts))
        if not counts:
            return 0
        return Counter(counts).most_common(1)[0][0]

    def _parse_rows(
        self, lines: List[str], delimiter: str, expected_cols: int
    ) -> Tuple[List[List[str]], int]:
        sep = re.escape(delimiter) if delimiter != r"\s+" else r"\s+"
        records: List[List[str]] = []
        rejected = 0

        for lineno, line in enumerate(lines, start=1):
            raw_parts = re.split(sep, line.strip())
            parts = [p.strip() for p in raw_parts if p.strip() != ""]

            if not parts:
                rejected += 1
                continue

            if len(parts) < expected_cols:
                parts.extend(["nan"] * (expected_cols - len(parts)))
            elif len(parts) > expected_cols:
                parts = parts[:expected_cols]

            numeric_count = sum(1 for p in parts if self._NUMERIC_RE.match(p.strip()))
            if numeric_count == 0:
                rejected += 1
                continue

            records.append(parts)

        return records, rejected

    def _apply_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.channel_names and len(self.config.channel_names) == len(df.columns):
            df.columns = self.config.channel_names
        return df

    def _extract_timestamps(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        ts_col_idx = self.config.timestamp_col

        if ts_col_idx is None:
            for col in df.columns:
                try:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    if vals.is_monotonic_increasing or vals.diff().dropna().gt(0).mean() > 0.8:
                        ts_col_idx = df.columns.get_loc(col)
                        break
                except Exception:
                    continue

        if ts_col_idx is not None:
            col_name = df.columns[ts_col_idx] if isinstance(ts_col_idx, int) else ts_col_idx
            ts = pd.to_numeric(df[col_name], errors="coerce")
            ts = (ts - ts.iloc[0]).astype(float)
            df = df.drop(columns=[col_name])
            return df, ts

        logger.warning("No timestamp column found; generating synthetic index.")
        n = len(df)
        sr = self.config.sampling_rate or 256.0
        ts = pd.Series(np.arange(n) / sr, name="timestamp_s")
        return df, ts

    def _coerce_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _handle_missing(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        missing_before = int(df.isna().sum().sum())
        strategy = self.config.fill_strategy

        if strategy == "interpolate":
            df = df.interpolate(method="linear", limit_direction="both")
        elif strategy == "ffill":
            df = df.ffill().bfill()
        elif strategy == "drop":
            df = df.dropna()
        else:
            df = df.interpolate(method="linear", limit_direction="both")

        missing_after = int(df.isna().sum().sum())
        return df, missing_before - missing_after

    def _infer_sampling_rate(self, index: pd.Index) -> float:
        if self.config.sampling_rate:
            return self.config.sampling_rate
        try:
            diffs = np.diff(index.to_numpy(dtype=float))
            median_dt = float(np.median(diffs[diffs > 0]))
            if median_dt > 0:
                return round(1.0 / median_dt, 4)
        except Exception:
            pass
        return 256.0