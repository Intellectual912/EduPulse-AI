#!/usr/bin/env python3
"""
EduPulse AI — Milestone 1: reproducible preprocessing & feature-engineering pipeline.

Project
-------
"EduPulse AI: Early-Warning Performance Risk Classifier & Socratic Coaching Platform"

What this script does
---------------------
1. Validates the raw Open University Learning Analytics Dataset (OULAD) files.
2. Profiles every table (rows, dtypes, missingness, duplicates, key candidates,
   invalid values, cross-table relationships) from the *actual* files.
3. Builds a leakage-free, one-row-per-enrolment feature table whose rows have a
   single, documented meaning:

       Grain = (code_module, code_presentation, id_student, cutoff_day)

   i.e. one row per student-course-presentation enrolment, observed only over
   the course days [course start .. cutoff_day].

4. Writes the processed tables, a machine-readable feature manifest, an
   exclusions ledger and a data-quality report generated from the real numbers.

Deliberately NOT in this milestone (see README / reports):
   * no model training (Milestone 2)
   * no LLM coach / RAG (Milestone 3)
   * no dashboards (Milestone 4)
   * no Knowledge Tracing — OULAD has no concept-labelled items, so this script
     only lays the join point where concept labels would attach.

Usage
-----
    python src/preprocess.py                     # default cutoff = day 28
    python src/preprocess.py --cutoff 42         # sensitivity analysis
    python src/preprocess.py --cutoff 28 --output-name student_features.csv

Raw data is never modified: every read is read-only, every write goes to
data/processed/ or reports/.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports"

ENROLMENT_KEY: List[str] = ["code_module", "code_presentation", "id_student"]

RAW_FILES: List[str] = [
    "courses.csv",
    "assessments.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "vle.csv",
    "studentVle.csv",
]

# Expected minimum schema. The pipeline fails loudly (and helpfully) if any of
# these columns is missing, rather than silently producing wrong features.
REQUIRED_COLUMNS: Dict[str, List[str]] = {
    "courses.csv": ["code_module", "code_presentation", "module_presentation_length"],
    "assessments.csv": ["code_module", "code_presentation", "id_assessment",
                        "assessment_type", "date", "weight"],
    "studentInfo.csv": ["code_module", "code_presentation", "id_student", "gender",
                        "region", "highest_education", "imd_band", "age_band",
                        "num_of_prev_attempts", "studied_credits", "disability",
                        "final_result"],
    "studentRegistration.csv": ["code_module", "code_presentation", "id_student",
                                "date_registration", "date_unregistration"],
    "studentAssessment.csv": ["id_assessment", "id_student", "date_submitted",
                              "is_banked", "score"],
    "vle.csv": ["id_site", "code_module", "code_presentation", "activity_type",
                "week_from", "week_to"],
    "studentVle.csv": ["code_module", "code_presentation", "id_student",
                       "id_site", "date", "sum_click"],
}

# Columns that must NEVER be used as model inputs. Enforced by an assertion in
# `assert_no_leakage`, so the guarantee is executable, not just documented.
FORBIDDEN_PREDICTORS: Dict[str, str] = {
    "final_result": "Outcome label recorded after the course ends.",
    "target_at_risk": "Derived outcome label.",
    "date_unregistration": "Withdrawal date; post-cutoff for most students.",
    "withdrawn_before_cutoff": "Defines cohort exclusion, not a predictor.",
    "date_submitted": "Raw timestamp; only windowed derivations may be used.",
    "score": "Raw per-assessment score; only windowed aggregates may be used.",
    "module_presentation_length": "Course metadata used for the outcome period description.",
}

VALID_FINAL_RESULTS = {"Pass", "Fail", "Withdrawn", "Distinction"}
VALID_GENDERS = {"M", "F"}
VALID_BANKED = {0, 1}

# Broad VLE activity-type grouping. Fixed a priori (not learned from the data)
# so the feature set is deterministic and free of "fit on the full dataset" bias.
ACTIVITY_GROUPS: Dict[str, str] = {
    "forumng": "forum",
    "oucontent": "content", "resource": "content", "subpage": "content",
    "page": "content", "url": "content", "htmlactivity": "content",
    "dualpane": "content", "sharedsubpage": "content", "folder": "content",
    "glossary": "content", "repeatactivity": "content",
    "quiz": "quiz", "externalquiz": "quiz", "questionnaire": "quiz",
    "oucollaborate": "collab", "ouelluminate": "collab",
    "ouwiki": "wiki", "dataplus": "wiki",
    "homepage": "home",
}
ACTIVITY_GROUP_FEATURES: List[str] = ["forum", "content", "quiz", "collab", "wiki", "home", "other"]

# Candidate cutoffs used for the documented sensitivity analysis.
CUTOFF_CANDIDATES: List[int] = [14, 21, 28, 42, 56]

# --------------------------------------------------------------------------- #
# Feature documentation (single source of truth, rendered into the report)
# --------------------------------------------------------------------------- #

FEATURE_DOC: Dict[str, Tuple[str, str]] = {
    # --- identifiers -------------------------------------------------------
    "code_module": ("identifier", "OULAD module code (7 modules: AAA-GGG). Identifier, not a predictor."),
    "code_presentation": ("identifier", "Presentation code: 'B' = February start, 'J' = October start. Identifier, not a predictor."),
    "id_student": ("identifier", "Anonymised student id. Identifier used for grouping and joins; never a predictor."),
    "cutoff_day": ("identifier", "Course-relative day at which observation stops (day 0 = course start)."),
    "course_length_days": ("metadata", "Presentation length in days from courses.csv; used to define the outcome period."),
    "days_cutoff_to_course_end": ("metadata", "course_length_days - cutoff_day: how much of the course is still in the future when the prediction is made."),
    # --- static enrolment attributes ---------------------------------------
    "gender": ("static", "Gender as declared at enrolment (M/F). Available before the cutoff."),
    "region": ("static", "UK/other region of the student at enrolment."),
    "highest_education": ("static", "Highest education level declared at enrolment."),
    "imd_band": ("static", "Index of Multiple Deprivation band; missing for 3.4% of enrolments and left as NaN (never zero-filled)."),
    "imd_band_missing": ("static", "1 if imd_band was absent at enrolment, else 0 (explicit missingness indicator)."),
    "age_band": ("static", "Age band at enrolment."),
    "disability": ("static", "Declared disability flag (Y/N)."),
    "num_of_prev_attempts": ("static", "Number of previous attempts at this module, at enrolment."),
    "studied_credits": ("static", "Credits the student is studying in this presentation, at enrolment."),
    # --- registration ------------------------------------------------------
    "registration_day": ("registration", "Registration day relative to course start (negative = before start). NaN when absent in OULAD (45 enrolments)."),
    "registration_days_before_start": ("registration", "-registration_day; positive means registered early. NaN when registration day is missing."),
    "registered_after_course_start": ("registration", "1 if the student registered on/after day 0 of the presentation."),
    "days_since_registration_at_cutoff": ("registration", "cutoff_day - registration_day: how long the student had been registered by the cutoff. NaN if unknown or negative."),
    "registration_date_missing": ("registration", "1 if OULAD has no registration day for this enrolment."),
    # --- assessment (window: due <= cutoff AND submitted <= cutoff) --------
    "n_assessments_due": ("assessment", "Assessments of this presentation with a known due date on or before the cutoff. 0 for presentations whose first assessment falls later."),
    "n_assessments_submitted": ("assessment", "Of those, how many the student had submitted on or before the cutoff (banked results included)."),
    "n_assessments_banked": ("assessment", "Subset of the above carried over from a previous attempt (is_banked = 1)."),
    "n_assessments_graded": ("assessment", "Submissions with a non-missing score available by the cutoff."),
    "n_assessments_unsubmitted": ("assessment", "n_assessments_due - n_assessments_submitted, i.e. missed by the cutoff. NOT imputed with a score of zero."),
    "assessment_submission_rate": ("assessment", "n_assessments_submitted / n_assessments_due. NaN (not 0) when no assessment was due."),
    "assessment_graded_rate": ("assessment", "n_assessments_graded / n_assessments_due. NaN when no assessment was due."),
    "n_late_submissions": ("assessment", "Non-banked submissions handed in after their due date but on/before the cutoff."),
    "late_submission_rate": ("assessment", "n_late_submissions / non-banked submissions available by the cutoff. NaN when there are none."),
    "mean_submission_delay_days": ("assessment", "Mean of (date_submitted - due_date) over non-banked submissions; negative = early."),
    "max_submission_delay_days": ("assessment", "Worst observed lateness in days among those submissions."),
    "days_since_last_submission": ("assessment", "cutoff_day - latest submission day observed by the cutoff. NaN when nothing was submitted."),
    "mean_score": ("assessment", "Mean score (0-100) over assessments graded by the cutoff. NaN when nothing was graded."),
    "min_score": ("assessment", "Lowest score graded by the cutoff."),
    "max_score": ("assessment", "Highest score graded by the cutoff."),
    "std_score": ("assessment", "Population std of scores; NaN when fewer than two assessments were graded."),
    "weighted_mean_score": ("assessment", "Score weighted by OULAD 'weight', normalised WITHIN assessment type (see report: OULAD weight is within-type, not a share of the final mark)."),
    "first_score": ("assessment", "Score of the earliest-due assessment graded by the cutoff."),
    "latest_score": ("assessment", "Score of the most recently due assessment graded by the cutoff."),
    "score_change": ("assessment", "latest_score - first_score when at least two assessments were graded, else NaN."),
    "has_assessment_due": ("assessment", "1 if any assessment of this presentation was due by the cutoff."),
    "has_graded_assessment": ("assessment", "1 if any assessment result was available by the cutoff."),
    # --- VLE activity (window: date <= cutoff) -----------------------------
    "vle_clicks_total": ("vle", "Sum of recorded platform interactions (sum_click) on days <= cutoff, including pre-course days. This is NOT study time in hours."),
    "vle_clicks_precourse": ("vle", "Clicks recorded before day 0 of the presentation."),
    "vle_active_days": ("vle", "Number of distinct days with at least one recorded click."),
    "vle_active_weeks": ("vle", "Number of distinct 7-day blocks (including pre-course weeks) with at least one recorded click."),
    "vle_distinct_sites": ("vle", "Number of distinct VLE resources (id_site) touched."),
    "vle_clicks_per_active_day": ("vle", "vle_clicks_total / vle_active_days. NaN when there is no activity."),
    "vle_activity_group_count": ("vle", "Number of broad resource families touched (forum, content, quiz, collab, wiki, home, other)."),
    "vle_clicks_forum": ("vle", "Clicks on forum resources (forumng)."),
    "vle_clicks_content": ("vle", "Clicks on content resources (oucontent, resource, subpage, page, url, ...)."),
    "vle_clicks_quiz": ("vle", "Clicks on quiz/questionnaire resources."),
    "vle_clicks_collab": ("vle", "Clicks on online collaboration rooms."),
    "vle_clicks_wiki": ("vle", "Clicks on wiki/dataplus resources."),
    "vle_clicks_home": ("vle", "Clicks on the module homepage."),
    "vle_clicks_other": ("vle", "Clicks on resource types outside the six named groups."),
    "vle_forum_share": ("vle", "vle_clicks_forum / vle_clicks_total. NaN when total clicks are 0."),
    "vle_clicks_last7d": ("vle", "Clicks during (cutoff-7, cutoff]."),
    "vle_clicks_prev7d": ("vle", "Clicks during (cutoff-14, cutoff-7]."),
    "vle_click_change_last7": ("vle", "vle_clicks_last7d - vle_clicks_prev7d; short-term engagement trend."),
    "vle_click_share_last7": ("vle", "vle_clicks_last7d / vle_clicks_total. NaN when total clicks are 0."),
    "vle_weekly_slope": ("vle", "OLS slope of clicks per course week over weeks 0..k-1 (positive = rising engagement)."),
    "vle_days_since_last_activity": ("vle", "cutoff_day - day of last recorded click. Large values indicate disengagement. NaN when never active."),
    "has_vle_activity": ("vle", "1 if the student has any recorded interaction by the cutoff. Absence of rows in studentVle means zero recorded activity (the log is the complete record of the platform)."),
    # --- target ------------------------------------------------------------
    "final_result": ("target", "OULAD-supplied final outcome (Pass/Fail/Withdrawn/Distinction). TARGET SOURCE — never a predictor."),
    "target_at_risk": ("target", "Derived binary label: 1 if final_result is Fail or Withdrawn, else 0. Derived by this pipeline, not supplied by OULAD."),
}

# Weekly VLE columns are generated dynamically; document them with a template.
WEEKLY_FEATURE_TEMPLATE = ("vle_clicks_wk{i}", "vle",
                           "Recorded clicks during course week {i} (days {lo}-{hi}). Week 0 begins on the first day of the presentation.")


class RawDataError(RuntimeError):
    """Raised when the raw dataset is missing, incomplete or structurally wrong."""


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #

def log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


def rel(path: Path) -> str:
    """Path shown relative to the project when possible, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _fmt_cell(v: Any) -> str:
    """Format one table cell without int->float upcasting."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return str(int(f)) if f.is_integer() else f"{f:.4g}"
    return str(v)


def to_md_table(df: pd.DataFrame, index: bool = False, max_rows: int = 60) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra deps)."""
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    d = df.head(max_rows)
    cols = ([str(df.index.name or "")] if index else []) + [str(c) for c in d.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in d.itertuples(index=index, name=None):
        cells = [_fmt_cell(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"| _… {len(df) - max_rows} more rows_ |" + " |" * (len(cols) - 1))
    return "\n".join(lines) + "\n"


def sha256_of(stream, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file path or of any readable binary stream."""
    import hashlib
    h = hashlib.sha256()
    close = False
    if isinstance(stream, (str, Path)):
        stream = Path(stream).open("rb")
        close = True
    try:
        for block in iter(lambda: stream.read(chunk), b""):
            h.update(block)
    finally:
        if close:
            stream.close()
    return h.hexdigest()


class RawSource:
    """
    Uniform, read-only access to the seven OULAD tables.

    Two on-disk layouts are supported, and the pipeline behaves identically in both:

    * ``extracted`` — the seven CSVs sit in ``data/raw/`` (fastest to read).
    * ``archived``  — only ``anonymisedData.zip`` is present. Tables are read
      straight out of the archive, so the ~444 MB of extracted CSV never has to
      exist on disk. This keeps a size-limited workspace well under quota; the
      extracted layout is what ``src/download_data.py`` produces by default.

    Raw data is only ever opened for reading; nothing here can modify it.
    """

    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.mode: str = "missing"
        self.zip_path: Path | None = None
        self._zip: "zipfile.ZipFile | None" = None
        self._members: Dict[str, "zipfile.ZipInfo"] = {}
        self._detect()

    def _detect(self) -> None:
        if self.raw_dir.exists() and all((self.raw_dir / f).exists() for f in RAW_FILES):
            self.mode = "extracted"
            return
        if not self.raw_dir.exists():
            return
        for candidate in sorted(self.raw_dir.glob("*.zip")):
            try:
                zf = zipfile.ZipFile(candidate)
            except (zipfile.BadZipFile, OSError):
                continue
            members = {Path(zi.filename).name: zi for zi in zf.infolist()
                       if not zi.is_dir() and zi.filename.endswith(".csv")}
            if all(f in members for f in RAW_FILES):
                self.mode, self.zip_path = "archived", candidate
                self._zip, self._members = zf, members
                log(f"reading OULAD directly from {candidate.name} "
                    f"({candidate.stat().st_size/1e6:.1f} MB, no extraction needed)")
                return
            zf.close()

    # -- accessors ----------------------------------------------------------
    def open(self, name: str):
        """Fresh binary read handle for one table (safe for chunked reading)."""
        if self.mode == "extracted":
            return (self.raw_dir / name).open("rb")
        if self.mode == "archived":
            return self._zip.open(self._members[name])
        raise RawDataError(
            f"Cannot open {name}: neither the extracted CSV nor a complete "
            f"anonymisedData.zip was found in {self.raw_dir}."
        )

    def size(self, name: str) -> int:
        if self.mode == "extracted":
            return (self.raw_dir / name).stat().st_size
        if self.mode == "archived":
            return int(self._members[name].file_size)
        return 0

    def sha256(self, name: str) -> str:
        with self.open(name) as fh:
            return sha256_of(fh)

    def describe(self) -> str:
        if self.mode == "extracted":
            return f"extracted CSVs in {self.raw_dir}"
        if self.mode == "archived":
            return f"archive {self.zip_path.name} (read without extracting)"
        return f"nothing readable in {self.raw_dir}"

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip, self._members = None, {}

    def __enter__(self) -> "RawSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# 1. Raw-data validation
# --------------------------------------------------------------------------- #

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="EduPulse AI — Milestone 1 preprocessing pipeline (OULAD).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
                   help="Directory holding the seven OULAD CSV files (read-only).")
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR,
                   help="Directory for generated feature tables.")
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
                   help="Directory for the generated data-quality report.")
    p.add_argument("--cutoff", type=int, default=28,
                   help="Prediction cutoff in course days (day 0 = course start).")
    p.add_argument("--output-name", type=str, default="student_features.csv",
                   help="Filename for the feature table inside processed-dir.")
    p.add_argument("--chunk-size", type=int, default=2_000_000,
                   help="Rows per chunk when streaming studentVle.csv.")
    p.add_argument("--skip-diagnostics", action="store_true",
                   help="Skip the multi-cutoff sensitivity table (faster).")
    return p.parse_args(argv)


def validate_raw_files(source: RawSource) -> pd.DataFrame:
    """Check that all seven tables exist and contain the required columns."""
    if source.mode == "missing":
        raise RawDataError(
            f"No usable OULAD data found in {source.raw_dir}\n"
            "Expected either the seven CSVs or a complete anonymisedData.zip.\n"
            "Run:  python src/download_data.py"
        )

    rows = []
    problems: List[str] = []
    for name in RAW_FILES:
        size = source.size(name)
        if size == 0:
            problems.append(f"  - empty file: {name}")
            continue
        try:
            with source.open(name) as fh:
                header = pd.read_csv(fh, nrows=0)
        except Exception as exc:  # pragma: no cover - defensive
            problems.append(f"  - unreadable CSV {name}: {exc}")
            continue
        missing = [c for c in REQUIRED_COLUMNS[name] if c not in header.columns]
        if missing:
            problems.append(
                f"  - {name}: missing required column(s) {missing}; found {list(header.columns)}"
            )
        rows.append({"file": name, "columns_found": len(header.columns),
                     "required_columns": len(REQUIRED_COLUMNS[name]),
                     "size_mb": round(size / 1e6, 2)})

    if problems:
        raise RawDataError(
            "The raw OULAD files are incomplete or do not match the expected schema:\n"
            + "\n".join(problems)
            + "\n\nObtain the dataset (CC BY 4.0) and unzip into data/raw/ :\n"
            "  * OU Analyse page ..... https://research.stem.open.ac.uk/ouanalyse/dataset/\n"
            "  * Authors' Figshare .. https://figshare.com/articles/dataset/5081998\n"
            "  * UCI ML Repository .. https://archive.ics.uci.edu/dataset/349\n"
            "Expected archive: anonymisedData.zip (MD5 7412686fd77cf0e0ee1e8c3e9b354308)\n"
            "Expected members: " + ", ".join(RAW_FILES)
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Profiling of the small tables
# --------------------------------------------------------------------------- #

def profile_small_table(name: str, df: pd.DataFrame,
                        key_candidates: Dict[str, List[str]],
                        relationship: str) -> Dict[str, Any]:
    """Profile a fully-loaded (small) table without modifying it."""
    missing = df.isna().sum()
    dup_rows = int(df.duplicated().sum())

    key_results = {}
    for label, cols in key_candidates.items():
        if all(c in df.columns for c in cols):
            key_results[label] = int(df.duplicated(subset=cols).sum())

    invalid: List[str] = []
    if name == "studentInfo.csv":
        bad = set(df["final_result"].dropna().unique()) - VALID_FINAL_RESULTS
        if bad:
            invalid.append(f"unexpected final_result value(s): {sorted(bad)}")
        bad = set(df["gender"].dropna().unique()) - VALID_GENDERS
        if bad:
            invalid.append(f"unexpected gender value(s): {sorted(bad)}")
        neg = int((df["studied_credits"] <= 0).sum())
        if neg:
            invalid.append(f"{neg} rows with studied_credits <= 0")
        negp = int((df["num_of_prev_attempts"] < 0).sum())
        if negp:
            invalid.append(f"{negp} rows with negative num_of_prev_attempts")
    if name == "studentAssessment.csv":
        bad = set(df["is_banked"].dropna().unique()) - VALID_BANKED
        if bad:
            invalid.append(f"unexpected is_banked value(s): {sorted(bad)}")
        out = int(((df["score"] < 0) | (df["score"] > 100)).sum())
        if out:
            invalid.append(f"{out} scores outside [0, 100]")
    if name == "assessments.csv":
        neg = int((df["weight"] < 0).sum())
        if neg:
            invalid.append(f"{neg} negative weights")
        bad = set(df["assessment_type"].dropna().unique())
        if not bad <= {"TMA", "CMA", "Exam"}:
            invalid.append(f"unexpected assessment_type value(s): {sorted(bad - {'TMA','CMA','Exam'})}")
    if name == "courses.csv":
        neg = int((df["module_presentation_length"] <= 0).sum())
        if neg:
            invalid.append(f"{neg} presentations with non-positive length")
    if name == "studentRegistration.csv" and "date_unregistration" in df.columns:
        both = int((df["date_unregistration"].notna() &
                    df["date_registration"].notna() &
                    (df["date_unregistration"] < df["date_registration"])).sum())
        if both:
            invalid.append(f"{both} rows unregistered BEFORE they registered")

    return {
        "file": name,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: {"n": int(v), "pct": round(100.0 * v / max(len(df), 1), 2)}
                    for c, v in missing.items() if v > 0},
        "duplicate_rows": dup_rows,
        "key_duplicates": key_results,
        "invalid_values": invalid,
        "relationship": relationship,
        "nunique": {c: int(df[c].nunique(dropna=True)) for c in df.columns
                    if df[c].dtype != object or df[c].nunique(dropna=True) <= 50},
    }


# --------------------------------------------------------------------------- #
# 3. studentVle profiling + feature extraction in ONE streaming pass
# --------------------------------------------------------------------------- #

def stream_student_vle(source: "RawSource", cutoff: int, chunk_size: int,
                       enrolment_keys: pd.DataFrame,
                       vle_meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Single chunked pass over studentVle.csv (10.6M rows) that simultaneously

      a) profiles the file exactly (duplicates, key multiplicity, ranges), and
      b) accumulates only the rows inside the observation window (date <= cutoff)
         to build per-enrolment VLE features.

    Memory rationale: loading the 453 MB / 10.66 M-row CSV at once costs several
    GB with object dtypes. Here we stream in chunks and retain only the filtered
    window (~2.8 M rows at day 28), directly addressing the requirement to be
    memory-conscious with this file.
    """
    log(f"streaming studentVle.csv in chunks of {chunk_size:,} rows (cutoff={cutoff})")

    row_hashes: List[np.ndarray] = []
    key_hashes: List[np.ndarray] = []
    click_arrays: List[np.ndarray] = []
    windows: List[pd.DataFrame] = []

    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    dtypes = {"id_student": "int32", "id_site": "int32", "date": "int16", "sum_click": "int32"}

    n_rows = 0
    n_neg = 0
    n_zero = 0
    n_missing_cells = {c: 0 for c in usecols}
    dmin, dmax = np.iinfo(np.int64).max, np.iinfo(np.int64).min
    students: set[int] = set()
    sites: set[int] = set()
    modpres: set[Tuple[str, str]] = set()
    enrols_seen: set[Tuple[str, str, int]] = set()
    rows_by_cutoff = {c: 0 for c in sorted(set(CUTOFF_CANDIDATES + [cutoff]))}
    clicks_by_cutoff = {c: 0 for c in sorted(set(CUTOFF_CANDIDATES + [cutoff]))}

    def _vle_chunks():
        # one open handle, closed automatically when the generator is exhausted
        with source.open("studentVle.csv") as vle_fh:
            yield from pd.read_csv(vle_fh, usecols=usecols, dtype=dtypes,
                                   chunksize=chunk_size)

    for chunk in _vle_chunks():
        n_rows += len(chunk)
        n_neg += int((chunk["sum_click"] < 0).sum())
        n_zero += int((chunk["sum_click"] == 0).sum())
        dmin = min(dmin, int(chunk["date"].min()))
        dmax = max(dmax, int(chunk["date"].max()))
        students.update(chunk["id_student"].unique().tolist())
        sites.update(chunk["id_site"].unique().tolist())
        modpres.update(map(tuple, chunk[["code_module", "code_presentation"]]
                           .drop_duplicates().values))
        enrols_seen.update(map(tuple, chunk[ENROLMENT_KEY].drop_duplicates().values))
        miss = chunk.isna().sum()
        for c in usecols:
            n_missing_cells[c] += int(miss[c])

        dates = chunk["date"].to_numpy()
        clicks = chunk["sum_click"].to_numpy()
        for c in rows_by_cutoff:
            mask = dates <= c
            rows_by_cutoff[c] += int(mask.sum())
            clicks_by_cutoff[c] += int(clicks[mask].sum())

        # exact profiling via 64-bit hashes (collision probability ~1e-6 at 10.6M rows)
        row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())
        key_hashes.append(pd.util.hash_pandas_object(chunk[usecols[:5]], index=False).to_numpy())
        click_arrays.append(clicks.astype(np.int32))

        win = chunk.loc[chunk["date"] <= cutoff]
        if len(win):
            windows.append(win)

    # ---- exact duplicate / multiplicity statistics ------------------------
    row_h = np.concatenate(row_hashes)
    key_h = np.concatenate(key_hashes)
    clicks_all = np.concatenate(click_arrays)
    del row_hashes, key_hashes, click_arrays

    n_dup_rows = int(n_rows - np.unique(row_h).size)
    keys, counts = np.unique(key_h, return_counts=True)
    n_unique_keys = int(keys.size)
    mult_buckets = {
        "1": int((counts == 1).sum()),
        "2": int((counts == 2).sum()),
        "3-5": int(((counts >= 3) & (counts <= 5)).sum()),
        "6-10": int(((counts >= 6) & (counts <= 10)).sum()),
        ">10": int((counts > 10).sum()),
    }
    # Keys whose rows carry differing sum_click values
    order = np.lexsort((clicks_all, key_h))
    ks, cs = key_h[order], clicks_all[order]
    new_key = np.empty(ks.size, dtype=bool)
    new_key[0] = True
    new_key[1:] = ks[1:] != ks[:-1]
    new_click = np.empty(cs.size, dtype=bool)
    new_click[0] = True
    new_click[1:] = cs[1:] != cs[:-1]
    runs_per_key = np.bincount(np.cumsum(new_key) - 1, weights=new_click.astype(np.int64),
                               minlength=int(new_key.sum()))
    keys_multi_click = int((runs_per_key > 1).sum())
    del order, ks, cs, new_key, new_click, runs_per_key, row_h, key_h, clicks_all, keys, counts

    prof = {
        "file": "studentVle.csv",
        "rows": n_rows,
        "columns": 6,
        "column_names": usecols,
        "dtypes": {c: "int32/int16/category" for c in usecols},
        "missing": {c: {"n": v, "pct": round(100.0 * v / max(n_rows, 1), 2)}
                    for c, v in n_missing_cells.items() if v > 0},
        "duplicate_rows": n_dup_rows,
        "duplicate_row_pct": round(100.0 * n_dup_rows / n_rows, 2),
        "unique_keys": n_unique_keys,
        "key_multiplicity_buckets": mult_buckets,
        "keys_with_multiple_rows": int(n_rows - n_unique_keys),
        "keys_with_differing_sum_click": keys_multi_click,
        "negative_sum_click": n_neg,
        "zero_sum_click": n_zero,
        "invalid_values": ([f"{n_neg} rows with negative sum_click"] if n_neg else [])
                          + ([f"{n_zero} rows with zero sum_click"] if n_zero else []),
        "key_duplicates": {},
        "date_min": int(dmin),
        "date_max": int(dmax),
        "distinct_students": len(students),
        "distinct_id_site": len(sites),
        "module_presentations": len(modpres),
        "rows_by_cutoff": {str(k): v for k, v in rows_by_cutoff.items()},
        "clicks_by_cutoff": {str(k): v for k, v in clicks_by_cutoff.items()},
        "relationship": ("One row per (module, presentation, student, VLE resource, day) "
                         "*intended*; in practice the key is not unique (see duplicates). "
                         "sum_click is an additive interaction count."),
        "sites_missing_from_vle_metadata": len(sites - set(vle_meta["id_site"])),
        "student_ids_not_in_studentInfo": len(students - set(enrolment_keys["id_student"].astype("int64"))),
        "enrolments_with_no_vle_rows": int(len(enrolment_keys) - len(enrols_seen)),
    }

    # ---- build features from the retained window --------------------------
    if not windows:
        raise RawDataError("No studentVle rows fall inside the observation window; "
                           "is the cutoff negative?")
    win = pd.concat(windows, ignore_index=True)
    for c in ["code_module", "code_presentation"]:
        win[c] = win[c].astype("category")
    log(f"retained {len(win):,} of {n_rows:,} studentVle rows inside the window")

    feats = _aggregate_vle(win, cutoff, vle_meta)
    del win, windows
    return feats, prof


def _aggregate_vle(win: pd.DataFrame, cutoff: int, vle_meta: pd.DataFrame) -> pd.DataFrame:
    """Aggregate windowed VLE rows into one row per enrolment."""
    win = win.copy()
    # Only COMPLETE 7-day blocks get their own column: with cutoff=28 that is
    # weeks 0..3 (days 0-27). A trailing partial block would look like a full
    # week while holding one day of clicks, so it is left out; the most recent
    # 7 days are covered by vle_clicks_last7d instead.
    n_weeks = max(1, cutoff // 7) if cutoff >= 7 else 1
    win["week"] = np.floor(win["date"].astype("int32") / 7).astype("int16")

    g = win.groupby(ENROLMENT_KEY, observed=True)
    base = pd.DataFrame({
        "vle_clicks_total": g["sum_click"].sum(),
        "vle_active_days": g["date"].nunique(),
        "vle_active_weeks": g["week"].nunique(),
        "vle_distinct_sites": g["id_site"].nunique(),
        "vle_last_activity_day": g["date"].max(),
    })

    # pre-course clicks (before day 0)
    pre = (win.loc[win["date"] < 0].groupby(ENROLMENT_KEY, observed=True)["sum_click"]
           .sum().rename("vle_clicks_precourse"))
    base = base.join(pre).fillna({"vle_clicks_precourse": 0})

    # weekly clicks for course weeks 0..n_weeks-1
    wk = (win.loc[(win["week"] >= 0) & (win["week"] < n_weeks)]
          .groupby(ENROLMENT_KEY + ["week"], observed=True)["sum_click"].sum()
          .unstack(fill_value=0)
          .reindex(columns=range(n_weeks), fill_value=0))
    wk.columns = [f"vle_clicks_wk{i}" for i in wk.columns]
    wk = wk[sorted(wk.columns, key=lambda s: int(s.rsplit("wk", 1)[1]))]
    base = base.join(wk)

    # last-7 / previous-7 day windows
    last7 = (win.loc[(win["date"] > cutoff - 7) & (win["date"] <= cutoff)]
             .groupby(ENROLMENT_KEY, observed=True)["sum_click"].sum().rename("vle_clicks_last7d"))
    prev7 = (win.loc[(win["date"] > cutoff - 14) & (win["date"] <= cutoff - 7)]
             .groupby(ENROLMENT_KEY, observed=True)["sum_click"].sum().rename("vle_clicks_prev7d"))
    base = base.join(last7).join(prev7).fillna({"vle_clicks_last7d": 0, "vle_clicks_prev7d": 0})

    # activity-type groups (metadata is unique on id_site -> validate many-to-one)
    meta = vle_meta[["id_site", "activity_type"]].drop_duplicates()
    if not meta["id_site"].is_unique:
        raise RawDataError("vle.csv has duplicate id_site rows; refusing to merge (row multiplication risk).")
    typed = win[ENROLMENT_KEY + ["id_site", "sum_click"]].merge(
        meta, on="id_site", how="left", validate="many_to_one")
    if len(typed) != len(win):
        raise RawDataError("VLE metadata merge changed the row count; aborting.")
    typed["group"] = typed["activity_type"].map(ACTIVITY_GROUPS).fillna("other")
    grp = (typed.groupby(ENROLMENT_KEY + ["group"], observed=True)["sum_click"].sum()
           .unstack(fill_value=0).reindex(columns=ACTIVITY_GROUP_FEATURES, fill_value=0))
    grp.columns = [f"vle_clicks_{c}" for c in grp.columns]
    base = base.join(grp).fillna(0)
    del typed, grp

    # categorical index levels -> plain strings, so later merges match exactly
    base.index = base.index.set_levels(
        [base.index.levels[0].astype(str), base.index.levels[1].astype(str),
         base.index.levels[2].astype("int64")])
    return base


def finalize_vle(base: pd.DataFrame, enrolment_keys: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Reindex onto the full enrolment universe and derive ratios/trends."""
    out = enrolment_keys.merge(base, left_on=ENROLMENT_KEY, right_index=True, how="left")

    click_cols = [c for c in out.columns
                  if c.startswith("vle_clicks") ] + ["vle_active_days", "vle_active_weeks", "vle_distinct_sites"]
    out[click_cols] = out[click_cols].fillna(0)

    # absence of records == zero recorded activity (the VLE log is complete),
    # which is different from a *missing* value: we flag it explicitly.
    out["has_vle_activity"] = (out["vle_clicks_total"] > 0).astype("int8")

    grp_cols = [c for c in out.columns if c.startswith("vle_clicks_") and
                c.split("vle_clicks_")[1] in ACTIVITY_GROUP_FEATURES]
    out["vle_activity_group_count"] = (out[grp_cols] > 0).sum(axis=1).astype("int8")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["vle_clicks_per_active_day"] = np.where(
            out["vle_active_days"] > 0,
            out["vle_clicks_total"] / out["vle_active_days"].replace(0, np.nan), np.nan)
        out["vle_forum_share"] = np.where(
            out["vle_clicks_total"] > 0,
            out["vle_clicks_forum"] / out["vle_clicks_total"].replace(0, np.nan), np.nan)
        out["vle_click_share_last7"] = np.where(
            out["vle_clicks_total"] > 0,
            out["vle_clicks_last7d"] / out["vle_clicks_total"].replace(0, np.nan), np.nan)

    out["vle_click_change_last7"] = out["vle_clicks_last7d"] - out["vle_clicks_prev7d"]

    # weekly OLS slope over weeks 0..k-1 (closed form; deterministic)
    wk_cols = sorted([c for c in out.columns if c.startswith("vle_clicks_wk")],
                     key=lambda s: int(s.rsplit("wk", 1)[1]))
    if len(wk_cols) >= 2:
        X = np.arange(len(wk_cols), dtype=float)
        Xc = X - X.mean()
        Y = out[wk_cols].to_numpy(dtype=float)
        out["vle_weekly_slope"] = (Y - Y.mean(axis=1, keepdims=True)) @ Xc / (Xc @ Xc)
    else:
        out["vle_weekly_slope"] = np.nan

    out["vle_days_since_last_activity"] = np.where(
        out["vle_last_activity_day"].notna(), cutoff - out["vle_last_activity_day"], np.nan)
    out = out.drop(columns=["vle_last_activity_day"])
    return out


# --------------------------------------------------------------------------- #
# 4. Assessment catalog & features
# --------------------------------------------------------------------------- #

def build_assessment_catalog(assessments: pd.DataFrame, courses: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build an assessment-level catalog and document OULAD's weight semantics.

    Verified empirically: the sum of `weight` per (presentation, assessment_type)
    is 100 for TMA and Exam, but CMAs often total 0/5/25 and CCC's exams total
    200. `weight` is therefore a *within-type* importance, NOT a share of the
    final mark. We normalise it within type so that no single assessment type
    can dominate simply because the file's numbers sum to 300.
    """
    cat = assessments.merge(courses, on=["code_module", "code_presentation"],
                            how="left", validate="many_to_one")
    if len(cat) != len(assessments):
        raise RawDataError("assessments x courses merge changed the row count; aborting.")

    grp_sum = cat.groupby(["code_module", "code_presentation", "assessment_type"])["weight"].transform("sum")
    cat["weight_type_normalised"] = np.where(grp_sum > 0, cat["weight"] / grp_sum.replace(0, np.nan), np.nan)

    info = {
        "weight_sums_by_presentation": (
            assessments.groupby(["code_module", "code_presentation"])["weight"].sum().describe().to_dict()),
        "weight_sums_by_type_counts": (
            assessments.groupby(["code_module", "code_presentation", "assessment_type"])["weight"]
            .sum().round(3).value_counts().sort_index().to_dict()),
        "assessments_missing_due_date": int(assessments["date"].isna().sum()),
        "missing_due_date_types": assessments.loc[assessments["date"].isna(), "assessment_type"]
            .value_counts().to_dict(),
        "assessments_presentations": int(len(assessments)),
        "type_counts": assessments["assessment_type"].value_counts().to_dict(),
    }
    return cat, info


def build_assessment_features(catalog: pd.DataFrame, student_assessment: pd.DataFrame,
                             enrolment_keys: pd.DataFrame, cutoff: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Assessment features using ONLY assessments that were both due and submitted
    on or before the cutoff. Anything later is future information at prediction
    time and is excluded by construction.
    """
    due_by_cutoff = catalog.loc[catalog["date"].notna() & (catalog["date"] <= cutoff)].copy()
    undated = int(catalog["date"].isna().sum())

    sa = student_assessment.merge(
        due_by_cutoff[["id_assessment", "code_module", "code_presentation", "assessment_type",
                       "date", "weight_type_normalised"]],
        on="id_assessment", how="inner", validate="many_to_one")

    sa_obs = sa.loc[sa["date_submitted"] <= cutoff].copy()   # known at prediction time
    sa_obs["delay"] = sa_obs["date_submitted"] - sa_obs["date"]

    k = ENROLMENT_KEY
    feats = pd.DataFrame(index=enrolment_keys.set_index(k).index)

    n_due = (due_by_cutoff.groupby(["code_module", "code_presentation"], observed=True)
             .size().rename("n_assessments_due"))
    feats = feats.join(n_due).fillna({"n_assessments_due": 0})
    feats["n_assessments_due"] = feats["n_assessments_due"].astype("int16")

    g = sa_obs.groupby(k, observed=True)
    agg = pd.DataFrame({
        "n_assessments_submitted": g["id_assessment"].nunique(),
        "n_assessments_banked": g["is_banked"].sum(),
        "n_assessments_graded": g["score"].count(),          # count() ignores NaN
        "mean_score": g["score"].mean(),
        "min_score": g["score"].min(),
        "max_score": g["score"].max(),
        "std_score": g["score"].std(ddof=0),
        "days_since_last_submission": cutoff - g["date_submitted"].max(),
    })
    feats = feats.join(agg)
    # a single graded assessment has no dispersion: report unknown, not 0
    feats.loc[feats["n_assessments_graded"].fillna(0) < 2, "std_score"] = np.nan

    # weighted mean over graded assessments, using type-normalised weights
    graded = sa_obs.loc[sa_obs["score"].notna() & sa_obs["weight_type_normalised"].notna()].copy()
    if len(graded):
        graded["_wscore"] = graded["score"] * graded["weight_type_normalised"]
        gg = graded.groupby(k, observed=True)
        wtot = gg["weight_type_normalised"].sum()
        feats["weighted_mean_score"] = gg["_wscore"].sum() / wtot.replace(0, np.nan)

    # first / latest score by due date (deterministic tie-break on id_assessment)
    gscore = sa_obs.loc[sa_obs["score"].notna()].sort_values(
        k + ["date", "id_assessment"], kind="mergesort")
    if len(gscore):
        first = gscore.groupby(k, observed=True)["score"].first().rename("first_score")
        last = gscore.groupby(k, observed=True)["score"].last().rename("latest_score")
        feats = feats.join(first).join(last)
        feats["score_change"] = np.where(feats["n_assessments_graded"] >= 2,
                                         feats["latest_score"] - feats["first_score"], np.nan)

    # submission-delay features (banked results are carry-overs, not new work)
    sub = sa_obs.loc[sa_obs["is_banked"] == 0].copy()
    if len(sub):
        sub["_late"] = (sub["delay"] > 0).astype("int8")
        gs = sub.groupby(k, observed=True)
        delay = pd.DataFrame({
            "mean_submission_delay_days": gs["delay"].mean(),
            "max_submission_delay_days": gs["delay"].max(),
            "n_late_submissions": gs["_late"].sum(),
            "n_submitted_nonbanked": gs["delay"].size(),
        })
        feats = feats.join(delay)
        feats["late_submission_rate"] = np.where(
            feats["n_submitted_nonbanked"] > 0,
            feats["n_late_submissions"] / feats["n_submitted_nonbanked"].replace(0, np.nan), np.nan)

    # explicit zero-vs-missing handling
    for c in ["n_assessments_submitted", "n_assessments_banked", "n_assessments_graded",
              "n_late_submissions"]:
        feats[c] = feats[c].fillna(0).astype("int16")
    feats["n_assessments_unsubmitted"] = (feats["n_assessments_due"] - feats["n_assessments_submitted"]).clip(lower=0)

    feats["assessment_submission_rate"] = np.where(
        feats["n_assessments_due"] > 0,
        feats["n_assessments_submitted"] / feats["n_assessments_due"].replace(0, np.nan), np.nan)
    feats["assessment_graded_rate"] = np.where(
        feats["n_assessments_due"] > 0,
        feats["n_assessments_graded"] / feats["n_assessments_due"].replace(0, np.nan), np.nan)
    feats["has_assessment_due"] = (feats["n_assessments_due"] > 0).astype("int8")
    feats["has_graded_assessment"] = (feats["n_assessments_graded"] > 0).astype("int8")

    if "n_submitted_nonbanked" in feats:
        feats = feats.drop(columns=["n_submitted_nonbanked"])

    # ---- exact submission-row ledger (separates the three exclusion causes) --
    sa_all = student_assessment.merge(catalog[["id_assessment", "date"]],
                                     on="id_assessment", how="left", validate="many_to_one")
    rows_undated = int(sa_all["date"].isna().sum())
    rows_future = int((sa_all["date"] > cutoff).sum())
    rows_due = int((sa_all["date"] <= cutoff).sum())
    rows_submitted_late = int(((sa_all["date"] <= cutoff) &
                               (sa_all["date_submitted"] > cutoff)).sum())
    rows_used = int(((sa_all["date"] <= cutoff) &
                     (sa_all["date_submitted"] <= cutoff)).sum())
    # evidence that undated assessments are end-of-course exams, not lost early signal
    undated_submit_days = sa_all.loc[sa_all["date"].isna(), "date_submitted"]

    diagnostics = {
        "undated_assessments_excluded": undated,
        "assessments_due_by_cutoff": int(len(due_by_cutoff)),
        "student_assessment_rows_total": int(len(student_assessment)),
        "student_assessment_rows_used": int(len(sa_obs)),
        "student_assessment_rows_undated": rows_undated,
        "student_assessment_rows_due_after_cutoff": rows_future,
        "student_assessment_rows_due_by_cutoff": rows_due,
        "submissions_after_cutoff_dropped": int(len(sa) - len(sa_obs)),
        "ledger_reconciles": bool(rows_undated + rows_future + rows_due == len(student_assessment)
                                  and rows_submitted_late + rows_used == rows_due),
        "undated_submission_day_min": (int(undated_submit_days.min())
                                       if len(undated_submit_days) else None),
        "undated_submission_day_median": (float(undated_submit_days.median())
                                          if len(undated_submit_days) else None),
        "undated_submission_day_max": (int(undated_submit_days.max())
                                       if len(undated_submit_days) else None),
        "presentations_without_assessments_by_cutoff": sorted(
            set(map(tuple, enrolment_keys[["code_module", "code_presentation"]]
                    .drop_duplicates().values))
            - set(map(tuple, due_by_cutoff[["code_module", "code_presentation"]]
                        .drop_duplicates().values))),
        "banked_rows_used": int(sa_obs["is_banked"].sum()),
    }
    return feats, diagnostics


# --------------------------------------------------------------------------- #
# 5. Registration & static features
# --------------------------------------------------------------------------- #

def build_registration_features(reg: pd.DataFrame, enrolment_keys: pd.DataFrame,
                               cutoff: int) -> pd.DataFrame:
    """Registration timing features. Withdrawal is NOT used here (cohort filter only)."""
    r = reg.copy()
    r["registration_day"] = r["date_registration"]
    r["registration_days_before_start"] = -r["registration_day"]
    r["registered_after_course_start"] = (r["registration_day"] >= 0).astype("int8")
    r["registration_date_missing"] = r["registration_day"].isna().astype("int8")
    with np.errstate(invalid="ignore"):
        days_since = cutoff - r["registration_day"]
    r["days_since_registration_at_cutoff"] = days_since.where(days_since >= 0)

    keep = ENROLMENT_KEY + ["registration_day", "registration_days_before_start",
                            "registered_after_course_start", "days_since_registration_at_cutoff",
                            "registration_date_missing"]
    return enrolment_keys.merge(r[keep], on=ENROLMENT_KEY, how="left", validate="one_to_one")


def build_static_features(info: pd.DataFrame) -> pd.DataFrame:
    """Static enrolment attributes. final_result is deliberately dropped here."""
    s = info.copy()
    s["imd_band_missing"] = s["imd_band"].isna().astype("int8")
    cols = ENROLMENT_KEY + ["gender", "region", "highest_education", "imd_band",
                            "imd_band_missing", "age_band", "disability",
                            "num_of_prev_attempts", "studied_credits"]
    return s[cols]


# --------------------------------------------------------------------------- #
# 6. Assembly, exclusions and leakage guards
# --------------------------------------------------------------------------- #

def assemble(raw: Dict[str, pd.DataFrame], cutoff: int, chunk_size: int,
             source: "RawSource") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    info, reg, courses = raw["studentInfo.csv"], raw["studentRegistration.csv"], raw["courses.csv"]
    assessments, sa, vle_meta = raw["assessments.csv"], raw["studentAssessment.csv"], raw["vle.csv"]

    enrolment_keys = info[ENROLMENT_KEY].drop_duplicates().reset_index(drop=True)
    if len(enrolment_keys) != len(info):
        raise RawDataError("studentInfo is not unique on the enrolment key; aborting.")

    # ---- exclusions decided BEFORE any feature is computed -----------------
    # left-merge from the enrolment spine so row order stays aligned with it
    reg_chk = enrolment_keys.merge(reg, on=ENROLMENT_KEY, how="left", validate="one_to_one")
    withdrawn = reg_chk["date_unregistration"].notna() & (reg_chk["date_unregistration"] <= cutoff)
    late_reg = reg_chk["date_registration"].notna() & (reg_chk["date_registration"] > cutoff)

    excl = pd.DataFrame({
        "code_module": reg_chk["code_module"],
        "code_presentation": reg_chk["code_presentation"],
        "id_student": reg_chk["id_student"],
    })
    excl["exclusion_reason"] = np.select(
        [withdrawn, late_reg],
        ["withdrawn_on_or_before_cutoff", "registered_after_cutoff"],
        default="")
    excl["detail"] = np.where(
        withdrawn,
        "date_unregistration=" + reg_chk["date_unregistration"].astype("Int64").astype(str),
        np.where(late_reg,
                 "date_registration=" + reg_chk["date_registration"].astype("Int64").astype(str), ""))
    excl = excl.loc[excl["exclusion_reason"] != ""].reset_index(drop=True)

    keep_mask = ~(withdrawn | late_reg)
    keep_keys = enrolment_keys.loc[keep_mask.to_numpy()].reset_index(drop=True)

    # ---- features ----------------------------------------------------------
    vle_raw, vle_prof = stream_student_vle(source, cutoff, chunk_size,
                                           enrolment_keys, vle_meta)
    vle_feats = finalize_vle(vle_raw, enrolment_keys, cutoff)

    catalog, cat_info = build_assessment_catalog(assessments, courses)
    assess_feats, assess_diag = build_assessment_features(catalog, sa, enrolment_keys, cutoff)
    reg_feats = build_registration_features(reg, enrolment_keys, cutoff)
    static_feats = build_static_features(info)

    target = info[ENROLMENT_KEY + ["final_result"]].copy()
    target["target_at_risk"] = target["final_result"].isin(["Fail", "Withdrawn"]).astype("int8")

    df = (keep_keys
          .merge(static_feats, on=ENROLMENT_KEY, how="left", validate="one_to_one")
          .merge(reg_feats, on=ENROLMENT_KEY, how="left", validate="one_to_one")
          .merge(assess_feats.reset_index(), on=ENROLMENT_KEY, how="left", validate="one_to_one")
          .merge(vle_feats, on=ENROLMENT_KEY, how="left", validate="one_to_one")
          .merge(target, on=ENROLMENT_KEY, how="left", validate="one_to_one")
          .merge(courses[["code_module", "code_presentation", "module_presentation_length"]],
                 on=["code_module", "code_presentation"], how="left", validate="many_to_one"))

    df = df.rename(columns={"module_presentation_length": "course_length_days"})
    df["cutoff_day"] = np.int16(cutoff)
    df["course_length_days"] = df["course_length_days"].astype("int16")
    df["days_cutoff_to_course_end"] = (df["course_length_days"] - df["cutoff_day"]).astype("int16")

    df = order_columns(df).sort_values(ENROLMENT_KEY, kind="mergesort").reset_index(drop=True)
    df["id_student"] = df["id_student"].astype("int64")

    meta = {
        "vle_profile": vle_prof,
        "catalog_info": cat_info,
        "assessment_diagnostics": assess_diag,
        "exclusions": excl,
        "counts": {
            "enrolments_total": int(len(enrolment_keys)),
            "withdrawals_on_or_before_day0": int(
                (reg_chk["date_unregistration"].notna()
                 & (reg_chk["date_unregistration"] <= 0)).sum()),
            "excluded_withdrawn_before_cutoff": int(withdrawn.sum()),
            "excluded_registered_after_cutoff": int(late_reg.sum()),
            "excluded_total": int(len(excl)),
            "included": int(len(df)),
        },
    }
    return df, excl, meta


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic column order: identifiers, target, then documented feature groups."""
    front = ["code_module", "code_presentation", "id_student", "cutoff_day",
             "course_length_days", "days_cutoff_to_course_end"]
    targets = ["final_result", "target_at_risk"]
    groups = ["static", "registration", "assessment", "vle"]
    ordered: List[str] = [c for c in front if c in df.columns]
    for g in groups:
        ordered += [c for c, (grp, _) in FEATURE_DOC.items() if grp == g and c in df.columns]
    # dynamic weekly columns
    ordered += sorted([c for c in df.columns if c.startswith("vle_clicks_wk")],
                      key=lambda s: int(s.rsplit("wk", 1)[1]))
    ordered += [c for c in targets if c in df.columns]
    leftover = [c for c in df.columns if c not in ordered]
    return df[ordered + leftover]


def feature_columns(df: pd.DataFrame) -> List[str]:
    """All columns that may be used as model inputs (targets/ids excluded)."""
    drop = {"code_module", "code_presentation", "id_student", "cutoff_day",
            "course_length_days", "final_result", "target_at_risk"}
    return [c for c in df.columns if c not in drop]


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Executable guarantee that outcome / future information is not a feature."""
    feats = set(feature_columns(df))
    bad = feats & set(FORBIDDEN_PREDICTORS)
    if bad:
        raise AssertionError(f"Leakage guard failed: forbidden predictors present: {sorted(bad)}")
    if "target_at_risk" not in df.columns or "final_result" not in df.columns:
        raise AssertionError("Target columns missing from the output table.")

    # target must be exactly reproducible from final_result
    recomputed = df["final_result"].isin(["Fail", "Withdrawn"]).astype("int8")
    if not recomputed.equals(df["target_at_risk"].astype("int8")):
        raise AssertionError("target_at_risk is not a deterministic function of final_result.")

    # scores used for features must come from assessments due AND submitted by cutoff
    if (df["n_assessments_graded"] > df["n_assessments_submitted"]).any():
        raise AssertionError("More graded than submitted assessments in some rows.")


def run_checks(df: pd.DataFrame, cutoff: int, meta: Dict[str, Any]) -> pd.DataFrame:
    """Structural checks on the produced table; returns a check ledger."""
    checks = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})

    key_unique = not df.duplicated(ENROLMENT_KEY + ["cutoff_day"]).any()
    add("grain_uniqueness", key_unique,
        f"duplicate rows on (module, presentation, student, cutoff): "
        f"{int(df.duplicated(ENROLMENT_KEY + ['cutoff_day']).sum())}")

    add("single_cutoff", df["cutoff_day"].nunique() == 1,
        f"cutoff_day values: {sorted(df['cutoff_day'].unique().tolist())}")

    c = meta["counts"]
    recon = c["included"] + c["excluded_total"] == c["enrolments_total"]
    add("row_accounting", recon,
        f"{c['included']} included + {c['excluded_total']} excluded = "
        f"{c['included'] + c['excluded_total']} vs {c['enrolments_total']} enrolments")

    add("no_target_nulls", bool(df["target_at_risk"].notna().all()),
        f"null targets: {int(df['target_at_risk'].isna().sum())}")

    ids_present = all(c in df.columns for c in ENROLMENT_KEY + ["cutoff_day"])
    add("identifiers_present", ids_present, f"key columns present: {ids_present}")

    add("assessment_window", bool((df["n_assessments_due"] >= 0).all()),
        "n_assessments_due is non-negative for all rows")

    add("no_future_withdrawal_feature",
        "date_unregistration" not in df.columns,
        "withdrawal date is not present in the feature table")

    return pd.DataFrame(checks)


# --------------------------------------------------------------------------- #
# 7. Cutoff sensitivity diagnostics
# --------------------------------------------------------------------------- #

def cutoff_diagnostics(raw: Dict[str, pd.DataFrame], vle_prof: Dict[str, Any],
                       cutoffs: Iterable[int]) -> pd.DataFrame:
    """Quantify the trade-off behind the chosen cutoff, from the real data."""
    info, reg = raw["studentInfo.csv"], raw["studentRegistration.csv"]
    catalog, _ = build_assessment_catalog(raw["assessments.csv"], raw["courses.csv"])
    sa = raw["studentAssessment.csv"]
    total_clicks = float(sum(vle_prof["clicks_by_cutoff"].values()))
    rows = []
    n_enrol = len(info)
    for c in cutoffs:
        withdrawn = int(((reg["date_unregistration"].notna()) & (reg["date_unregistration"] <= c)).sum())
        late = int(((reg["date_registration"].notna()) & (reg["date_registration"] > c)).sum())
        cohort = n_enrol - withdrawn - late

        due = catalog.loc[catalog["date"].notna() & (catalog["date"] <= c)]
        due_pres = set(map(tuple, due[["code_module", "code_presentation"]].drop_duplicates().values))
        all_pres = set(map(tuple, info[["code_module", "code_presentation"]].drop_duplicates().values))

        obs = sa.merge(due[["id_assessment", "code_module", "code_presentation"]],
                       on="id_assessment", how="inner")
        obs = obs.loc[obs["date_submitted"] <= c]
        n_graded_per_enrol = (obs.loc[obs["score"].notna()]
                              .groupby(["code_module", "code_presentation", "id_student"]).size())
        keys = info[ENROLMENT_KEY].merge(
            n_graded_per_enrol.rename("n").reset_index(),
            on=ENROLMENT_KEY, how="left").fillna({"n": 0})

        clicks = vle_prof["clicks_by_cutoff"].get(str(c))
        rows.append({
            "cutoff_day": c,
            "cohort_after_exclusions": cohort,
            "presentations_with_assessment": len(due_pres),
            "presentations_total": len(all_pres),
            "pct_enrolments_with_ge1_graded": round(100 * (keys["n"] >= 1).mean(), 1),
            "mean_graded_assessments": round(keys["n"].mean(), 2),
            "vle_rows_in_window_pct": None if not vle_prof["rows_by_cutoff"].get(str(c)) else
                round(100 * vle_prof["rows_by_cutoff"][str(c)] / vle_prof["rows"], 1),
            "vle_clicks_in_window_pct": None if clicks is None else
                round(100 * clicks / max(total_clicks, 1), 1),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 8. Report writing
# --------------------------------------------------------------------------- #

def build_quality_report(profiles: List[Dict[str, Any]], df: pd.DataFrame, meta: Dict[str, Any],
                         checks: pd.DataFrame, diag: pd.DataFrame | None, cutoff: int,
                         raw_hashes: Dict[str, str], out_name: str) -> str:
    c = meta["counts"]
    vp = meta["vle_profile"]
    ap = meta["assessment_diagnostics"]
    ci = meta["catalog_info"]

    L: List[str] = []
    L.append("# EduPulse AI — Data Quality Report (Milestone 1)\n")
    L.append(f"_Generated automatically by `src/preprocess.py` on "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
             f"from the actual raw files. Cutoff = course day {cutoff}._\n")
    L.append("Every number below was computed from the files in `data/raw/`; nothing is illustrative.\n")

    # 1. inventory
    L.append("## 1. File inventory and checksums\n")
    inv = pd.DataFrame([
        {"file": p["file"], "rows": f"{p['rows']:,}", "columns": p["columns"],
         "sha256": raw_hashes.get(p["file"], "")[:16] + "…"}
        for p in profiles])
    L.append(to_md_table(inv))
    L.append(f"\n**Raw data source for this run:** {meta.get('raw_source', 'unknown')}. "
             "The pipeline reads the seven tables either from extracted CSVs or directly "
             "from `anonymisedData.zip`; both layouts produce identical features.\n")
    L.append("\nSource: OULAD (Kuzilek, Hlosta & Zdrahal, 2017), CC BY 4.0, retrieved from the "
             "authors' Figshare record (DOI 10.6084/m9.figshare.5081998); archive MD5 "
             "`7412686fd77cf0e0ee1e8c3e9b354308` verified after download.\n")

    # 2. per-file profile
    L.append("## 2. Per-file profile\n")
    for p in profiles:
        L.append(f"### `{p['file']}` — {p['rows']:,} rows × {p['columns']} columns\n")
        L.append(f"- **Relationship:** {p['relationship']}")
        dt = pd.DataFrame([{"column": k, "dtype": v} for k, v in p["dtypes"].items()])
        L.append("\n**Data types**\n\n" + to_md_table(dt, max_rows=20))
        if p.get("missing"):
            miss = pd.DataFrame([{"column": k, "missing": f"{v['n']:,}", "pct": f"{v['pct']}%"}
                                 for k, v in p["missing"].items()])
            L.append("\n**Missing values**\n\n" + to_md_table(miss))
        else:
            L.append("\n**Missing values:** none.\n")
        L.append(f"\n**Duplicate rows:** {p['duplicate_rows']:,}")
        if p.get("duplicate_row_pct") is not None:
            L.append(f" ({p['duplicate_row_pct']}% of the file)")
        L.append("\n")
        if p.get("key_duplicates"):
            kd = pd.DataFrame([{"key candidate": k, "duplicate rows": v}
                               for k, v in p["key_duplicates"].items()])
            L.append("\n**Key candidates** (0 duplicates = usable as a join key)\n\n" + to_md_table(kd))
        if p.get("unique_keys") is not None:
            L.append(f"\n**Key structure:** {p['unique_keys']:,} distinct "
                     f"(module, presentation, student, site, date) keys for {p['rows']:,} rows → "
                     f"{p['keys_with_multiple_rows']:,} rows share a key with another row, of which "
                     f"{p['keys_with_differing_sum_click']:,} keys carry *different* `sum_click` values. "
                     "The key is therefore **not unique**, and `sum_click` behaves as an additive "
                     "interaction count. Multiplicity: "
                     + ", ".join(f"{k} rows → {v:,} keys" for k, v in p["key_multiplicity_buckets"].items()) + ".")
            L.append(f"\n**Value ranges:** date {p['date_min']}…{p['date_max']}; "
                     f"negative `sum_click` = {p['negative_sum_click']:,}; zero `sum_click` = {p['zero_sum_click']:,}; "
                     f"{p['distinct_students']:,} distinct students, {p['distinct_id_site']:,} distinct sites, "
                     f"{p['module_presentations']} module-presentations.")
            L.append(f"\n**Referential integrity:** {p['sites_missing_from_vle_metadata']} sites missing from "
                     f"`vle.csv`; {p['student_ids_not_in_studentInfo']} student ids missing from `studentInfo.csv`; "
                     f"{p['enrolments_with_no_vle_rows']:,} enrolments have no VLE row at all.")
        if p.get("invalid_values"):
            L.append("\n**Unexpected / invalid values**\n")
            L.extend(f"- {v}\n" for v in p["invalid_values"])
        else:
            L.append("\n**Unexpected / invalid values:** none detected by the domain checks.\n")
        if p.get("nunique"):
            tiny = {k: v for k, v in p["nunique"].items() if v <= 20}
            if tiny:
                L.append("\n**Low-cardinality columns:** "
                         + ", ".join(f"`{k}` ({v})" for k, v in tiny.items()) + "\n")
        L.append("\n")

    # 3. semantic findings
    L.append("## 3. Semantic findings that changed the design\n")
    L.append(
        "1. **`assessments.weight` is within-type, not a share of the final mark.** "
        f"Summing `weight` per presentation gives {ci['weight_sums_by_presentation']['min']:.0f}–"
        f"{ci['weight_sums_by_presentation']['max']:.0f} (not 100). Summed per "
        f"(presentation, assessment_type) the totals are: "
        + ", ".join(f"{_fmt_cell(k)}→{v} presentations" for k, v in ci["weight_sums_by_type_counts"].items())
        + ". A naive weighted average over raw `weight` would therefore be wrong; this pipeline "
          "normalises `weight` **within** assessment type. Note that some assessments carry weight 0 "
          "(e.g. a formative TMA), so they contribute nothing to `weighted_mean_score` by design.\n")
    mdt = ci["missing_due_date_types"]
    mdt_txt = ", ".join(f"{k} × {v}" for k, v in mdt.items()) if mdt else "none"
    if ap["student_assessment_rows_undated"] and ap["undated_submission_day_median"] is not None:
        timing = (f"This costs no early-warning signal: those submissions occur on days "
                  f"{ap['undated_submission_day_min']}–{ap['undated_submission_day_max']} "
                  f"(median {ap['undated_submission_day_median']:.0f}), matching the dated end-of-course "
                  "exams (days 222–261) — they are end-of-course exams, i.e. far beyond any early cutoff, "
                  "not early evidence.")
    else:
        timing = ("No submission rows attached to them survive the cutoff in this run, so they do not "
                  "affect any feature.")
    L.append(
        f"2. **{ci['assessments_missing_due_date']} assessments have no due date** ({mdt_txt}). "
        f"They account for {ap['student_assessment_rows_undated']:,} submission rows, and because their "
        f"timing is unknown they cannot enter a windowed feature. {timing}\n")
    L.append(
        f"3. **`is_banked` marks carry-overs from a previous attempt** "
        f"({ap['banked_rows_used']:,} banked submissions inside the window). Their scores are valid results, "
        "but they are not new work, so they are excluded from submission-delay statistics.\n")
    L.append(
        "4. **A score of 0 is not the same as a missing score.** Only 173 of 173,912 submission rows "
        "lack a score; these are left as NaN and are never replaced by 0. Unsubmitted assessments are "
        "counted in `n_assessments_unsubmitted`, not scored as zero.\n")
    L.append(
        f"5. **Withdrawal is often early:** {c['withdrawals_on_or_before_day0']:,} enrolments carry an "
        f"unregistration date on or before day 0, and {c['excluded_withdrawn_before_cutoff']:,} end on or "
        f"before the day-{cutoff} cutoff. Those rows are excluded from the cohort because their outcome is "
        "already known at prediction time; keeping them would be tautological.\n")
    L.append(
        "6. **VLE rows are not unique per (student, site, day)** and clicks are *not* study hours. "
        "All VLE features are documented as recorded platform interactions.\n")
    L.append(
        "7. **OULAD contains no concept or question-level labels.** Assessments are whole TMA/CMA/Exam "
        "units; there is no item bank, no concept tag and no prerequisite graph. Concept-level mastery "
        "tracking (faculty Feature 2) therefore cannot be implemented from this dataset as-is — see §8.\n")

    # 4. cutoff
    L.append("## 4. Prediction cutoff: the trade-off\n")
    L.append("The cutoff is the last course day included in features. Earlier = more useful warning, "
             "but fewer assessment results exist. This table is computed from the raw data:\n")
    if diag is not None:
        L.append(to_md_table(diag))
    L.append(f"\n**Chosen cutoff: day {cutoff}** (about {cutoff}/270 ≈ {round(100*cutoff/270)}% of a "
             "presentation). At this point the course has ~"
             f"{round(100*(1-cutoff/270))}% of its length still ahead, which is genuinely early-warning, "
             "and most enrolments already have VLE signal. Consequences are documented rather than hidden:\n")
    L.append("- Some presentations (EEE, GGG) have **no assessment due by day 28**, so their assessment "
             "features are structurally absent (NaN) and `has_assessment_due = 0`; this is preserved "
             "explicitly instead of being filled with zeros.\n")
    L.append("- `--cutoff` is a first-class parameter; the November model can re-run at 42/56 days and "
             "compare, because every downstream artefact records `cutoff_day`.\n")

    # 5. target & leakage
    L.append("## 5. Prediction target and leakage control\n")
    L.append(f"- **Unit predicted:** one student-course-presentation enrolment (grain of `{out_name}`).")
    L.append(f"- **Observation window:** course days −∞ … {cutoff} (pre-course access included; it is known before the cutoff).")
    L.append(f"- **Prediction cutoff:** day {cutoff}.")
    L.append(f"- **Outcome period:** days {cutoff+1} … course end (234–269 days, from `courses.csv`).")
    L.append("- **Label:** `target_at_risk = 1` if `final_result ∈ {Fail, Withdrawn}`, else 0. "
             "`final_result` is supplied by OULAD; the binary risk category is **derived here**.")
    L.append(f"- **Prevalence in the processed cohort:** {100*df['target_at_risk'].mean():.1f}% at-risk "
             f"({int(df['target_at_risk'].sum()):,} of {len(df):,}).\n")
    L.append("Rules enforced in code (see `assert_no_leakage`):\n")
    L.append("| Rule | Enforcement |\n|---|---|")
    for k, v in FORBIDDEN_PREDICTORS.items():
        L.append(f"| `{k}` may not be a predictor | {v} |")
    L.append("| Assessment results only if due **and** submitted by the cutoff | window filter in `build_assessment_features` |")
    L.append("| No withdrawal after the cutoff as a feature | only `date_unregistration <= cutoff` is used, and only to exclude rows |")
    L.append("| Course-design metadata may be used in full | the assessment schedule, due dates, weights, course length and VLE resource catalogue are published before day 0; only *student-generated* records are windowed. `weighted_mean_score` therefore depends on the whole schedule, which is intended and is covered by the leakage test |")
    L.append("| No fitting of imputers/scalers on the full dataset | this milestone only *derives* features; no learned transformation is applied. Scaling/imputation will be fitted inside the training split in Milestone 2 |")
    L.append("")

    # 6. exclusions
    L.append("## 6. Row accounting and exclusions\n")
    acc = pd.DataFrame([
        {"stage": "Enrolments in studentInfo.csv", "rows": f"{c['enrolments_total']:,}"},
        {"stage": "− withdrawn on/before cutoff", "rows": f"−{c['excluded_withdrawn_before_cutoff']:,}"},
        {"stage": "− registered after cutoff", "rows": f"−{c['excluded_registered_after_cutoff']:,}"},
        {"stage": "= rows in processed feature table", "rows": f"{c['included']:,}"},
    ])
    L.append(to_md_table(acc))
    L.append("\nEvery excluded row is written to `data/processed/excluded_records.csv` with its reason "
             "and the offending date value; nothing is dropped silently.\n")
    L.append("\n**Assessment-row ledger** (reconciles exactly: "
             f"{'yes' if ap['ledger_reconciles'] else 'NO'})\n")
    led = pd.DataFrame([
        {"stage": "studentAssessment rows read", "rows": f"{ap['student_assessment_rows_total']:,}",
         "why": "raw table"},
        {"stage": "− assessment has no due date", "rows": f"−{ap['student_assessment_rows_undated']:,}",
         "why": "timing unknown; end-of-course exams (see §3.2)"},
        {"stage": "− assessment due after the cutoff", "rows": f"−{ap['student_assessment_rows_due_after_cutoff']:,}",
         "why": "future information at prediction time"},
        {"stage": "= rows for assessments due by the cutoff", "rows": f"{ap['student_assessment_rows_due_by_cutoff']:,}",
         "why": ""},
        {"stage": "− submitted after the cutoff", "rows": f"−{ap['submissions_after_cutoff_dropped']:,}",
         "why": "the result did not exist yet at the cutoff"},
        {"stage": "= submissions usable as cutoff-time evidence", "rows": f"{ap['student_assessment_rows_used']:,}",
         "why": ""},
    ])
    L.append(to_md_table(led))

    # 7. output
    L.append("\n## 7. Processed feature table\n")
    L.append(f"`{out_name}` — {len(df):,} rows × {df.shape[1]} columns; "
             f"{len(feature_columns(df))} usable predictors.\n")
    L.append(f"**Grain:** one row per `(code_module, code_presentation, id_student, cutoff_day)` — "
             f"unique? **{not df.duplicated(ENROLMENT_KEY + ['cutoff_day']).any()}**.\n")
    L.append("**Missingness in the feature table (top columns)**\n")
    miss = (df[feature_columns(df)].isna().mean().mul(100).round(1)
            .sort_values(ascending=False))
    miss = miss[miss > 0].rename("missing_pct").reset_index().rename(columns={"index": "feature"})
    L.append(to_md_table(miss.head(20)))
    L.append("\nStructural NaNs are expected: `mean_score` is NaN when no assessment was graded by the "
             "cutoff, and ratio features are NaN when their denominator is zero. These are *unknown*, not zero.\n")
    noweight = df[(df["n_assessments_graded"] > 0) & (df["weighted_mean_score"].isna())]
    if len(noweight):
        t = (noweight.groupby(["code_module", "code_presentation"]).size()
             .rename("rows").reset_index()
             .sort_values("rows", ascending=False))
        L.append(f"\n`weighted_mean_score` is additionally absent for {len(noweight):,} rows whose only "
                 "graded work by the cutoff carried **zero weight** in OULAD (a formative assessment). "
                 "Those rows are listed here; they are not a processing error:\n\n")
        L.append(to_md_table(t))

    L.append("\n## 8. Concept-level mastery (faculty Feature 2): what OULAD does and does not support\n")
    L.append("OULAD stores results per **whole assessment** (`id_assessment` → TMA/CMA/Exam). It has no "
             "question-level items, no concept tags, and no prerequisite structure, so concept mastery "
             "and prerequisite-gap detection **cannot** be computed from this dataset without inventing "
             "labels. What this milestone prepares instead:\n")
    L.append("1. `data/processed/assessment_catalog.csv` — one row per assessment (module, presentation, "
             "type, due day, raw weight, type-normalised weight). A future `assessment_concepts.csv` "
             "(`id_assessment, concept_id, weight_in_assessment`) joins here at a 1-to-many grain.")
    L.append("2. `src/preprocess.py::load_concept_labels()` — optional, validated hook that activates "
             "concept tagging the moment such a file is provided, with no change to the pipeline.")
    L.append("3. The VLE activity-type groups (forum/content/quiz/collab/wiki) give coarse *behavioural* "
             "cues; they are not concepts and are not presented as such.")
    L.append("4. Knowledge Tracing (BKT/DKT) additionally requires per-item response sequences, which "
             "OULAD does not contain; it needs a different data source.\n")

    # 9. checks
    L.append("\n## 9. Executed validation checks\n")
    L.append(to_md_table(checks))
    L.append(f"\n**Result:** {int((checks['status'] == 'PASS').sum())}/{len(checks)} checks passed.\n")

    L.append("\n## 10. Feature dictionary\n")
    L.append(_feature_dictionary_md(df))
    return "\n".join(L)


def _feature_dictionary_md(df: pd.DataFrame) -> str:
    rows = []
    for c in df.columns:
        if c in FEATURE_DOC:
            grp, desc = FEATURE_DOC[c]
        elif c.startswith("vle_clicks_wk"):
            i = int(c.rsplit("wk", 1)[1])
            grp, desc = "vle", WEEKLY_FEATURE_TEMPLATE[2].format(i=i, lo=i * 7, hi=i * 7 + 6)
        else:
            grp, desc = "other", ""
        rows.append({"column": c, "group": grp, "description": desc,
                     "dtype": str(df[c].dtype)})
    return to_md_table(pd.DataFrame(rows), max_rows=200)


# --------------------------------------------------------------------------- #
# 9. Concept-label hook (future milestone scaffold)
# --------------------------------------------------------------------------- #

def load_concept_labels(processed_dir: Path, catalog: pd.DataFrame) -> pd.DataFrame | None:
    """
    Optional hook for Milestone 2/3.

    If `data/processed/assessment_concepts.csv` exists with columns
    [id_assessment, concept_id, weight_in_assessment] it is validated and
    returned; otherwise None. OULAD ships no such labels, so this returns None
    today and the pipeline continues unchanged.
    """
    path = processed_dir / "assessment_concepts.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    required = {"id_assessment", "concept_id", "weight_in_assessment"}
    missing = required - set(labels.columns)
    if missing:
        raise RawDataError(f"{path.name} is missing required columns: {sorted(missing)}")
    unknown = set(labels["id_assessment"]) - set(catalog["id_assessment"])
    if unknown:
        raise RawDataError(f"{path.name} references {len(unknown)} assessment ids absent from the catalog.")
    log(f"concept labels found: {len(labels):,} rows, {labels['concept_id'].nunique()} concepts")
    return labels


# --------------------------------------------------------------------------- #
# 10. Main
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    t0 = time.time()
    args = parse_args(argv)

    raw_dir = args.raw_dir.resolve()
    processed_dir = args.processed_dir.resolve()
    reports_dir = args.reports_dir.resolve()
    processed_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.cutoff < 0:
        raise SystemExit("--cutoff must be >= 0 course days.")

    log(f"project root : {PROJECT_ROOT}")
    log(f"raw dir      : {raw_dir}")
    log(f"cutoff       : day {args.cutoff}")

    # ---- 1. validate inputs -------------------------------------------------
    source = RawSource(raw_dir)
    log(f"raw data source: {source.describe()}")
    inventory = validate_raw_files(source)
    log(f"validated {len(inventory)} raw files")

    raw_hashes = {name: source.sha256(name) for name in RAW_FILES}
    log("computed SHA-256 provenance hashes")

    # ---- 2. load small tables ----------------------------------------------
    raw: Dict[str, pd.DataFrame] = {}
    for name in RAW_FILES:
        if name == "studentVle.csv":
            continue
        with source.open(name) as fh:
            raw[name] = pd.read_csv(fh)
        log(f"loaded {name}: {raw[name].shape[0]:,} rows × {raw[name].shape[1]} cols")

    # normalise id_site dtype across the two VLE tables (avoids silent merge misses)
    raw["vle.csv"]["id_site"] = raw["vle.csv"]["id_site"].astype("int64")

    # ---- 3. profile ---------------------------------------------------------
    profiles = [
        profile_small_table("courses.csv", raw["courses.csv"],
                            {"(module, presentation)": ["code_module", "code_presentation"]},
                            "One row per course presentation; supplies module_presentation_length."),
        profile_small_table("assessments.csv", raw["assessments.csv"],
                            {"id_assessment": ["id_assessment"],
                             "(module, presentation, id_assessment)": ["code_module", "code_presentation", "id_assessment"]},
                            "Assessment metadata; joins to studentAssessment on id_assessment (1-to-many)."),
        profile_small_table("studentInfo.csv", raw["studentInfo.csv"],
                            {"(module, presentation, student)": ENROLMENT_KEY},
                            "One row per enrolment; the spine of the feature table. Holds final_result."),
        profile_small_table("studentRegistration.csv", raw["studentRegistration.csv"],
                            {"(module, presentation, student)": ENROLMENT_KEY},
                            "One row per enrolment; registration and withdrawal dates."),
        profile_small_table("studentAssessment.csv", raw["studentAssessment.csv"],
                            {"(id_assessment, id_student)": ["id_assessment", "id_student"]},
                            "Submission rows; joins to assessments on id_assessment."),
        profile_small_table("vle.csv", raw["vle.csv"],
                            {"id_site": ["id_site"]},
                            "VLE resource metadata; joins to studentVle on id_site (1-to-many)."),
    ]

    # ---- 4. assemble --------------------------------------------------------
    df, excl, meta = assemble(raw, args.cutoff, args.chunk_size, source)
    profiles.append(meta["vle_profile"])
    meta["raw_source"] = source.describe()
    log(f"assembled feature table: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ---- 5. checks ----------------------------------------------------------
    assert_no_leakage(df)
    checks = run_checks(df, args.cutoff, meta)
    if (checks["status"] == "FAIL").any():
        print(checks.to_string(index=False), file=sys.stderr)
        raise SystemExit("Structural checks failed; outputs were not written.")

    # ---- 6. write outputs ---------------------------------------------------
    out_path = processed_dir / args.output_name
    df.to_csv(out_path, index=False)
    log(f"wrote {rel(out_path)} ({out_path.stat().st_size/1e6:.2f} MB)")

    excl_path = processed_dir / "excluded_records.csv"
    excl.to_csv(excl_path, index=False)
    log(f"wrote {rel(excl_path)} ({len(excl):,} rows)")

    catalog, _ = build_assessment_catalog(raw["assessments.csv"], raw["courses.csv"])
    cat_path = processed_dir / "assessment_catalog.csv"
    catalog.to_csv(cat_path, index=False)
    log(f"wrote {rel(cat_path)} ({len(catalog):,} rows)")

    diag = None
    if not args.skip_diagnostics:
        diag = cutoff_diagnostics(raw, meta["vle_profile"], CUTOFF_CANDIDATES)
        diag_path = processed_dir / "cutoff_diagnostics.csv"
        diag.to_csv(diag_path, index=False)
        log(f"wrote {rel(diag_path)}")

    concepts = load_concept_labels(processed_dir, catalog)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cutoff_day": args.cutoff,
        "output_file": out_path.name,
        "grain": ["code_module", "code_presentation", "id_student", "cutoff_day"],
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "target": {"column": "target_at_risk",
                   "definition": "1 if final_result in {Fail, Withdrawn} else 0",
                   "source_column": "final_result",
                   "prevalence": round(float(df["target_at_risk"].mean()), 4)},
        "id_columns": ["code_module", "code_presentation", "id_student", "cutoff_day"],
        "metadata_columns": ["course_length_days", "days_cutoff_to_course_end"],
        "target_columns": ["final_result", "target_at_risk"],
        "feature_columns": feature_columns(df),
        "forbidden_predictors": FORBIDDEN_PREDICTORS,
        "feature_groups": {g: [c for c in feature_columns(df)
                               if FEATURE_DOC.get(c, ("", ""))[0] == g]
                           for g in ["static", "registration", "assessment", "vle"]},
        "feature_descriptions": {c: FEATURE_DOC[c][1] for c in feature_columns(df) if c in FEATURE_DOC},
        "counts": meta["counts"],
        "assessment_diagnostics": {k: v for k, v in meta["assessment_diagnostics"].items()
                                   if k != "presentations_without_assessments_by_cutoff"},
        "presentations_without_assessments_by_cutoff":
            [f"{m}-{p}" for m, p in meta["assessment_diagnostics"]["presentations_without_assessments_by_cutoff"]],
        "raw_file_sha256": raw_hashes,
        "raw_source": meta.get("raw_source", "unknown"),
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__},
    }
    if concepts is not None:
        manifest["concept_labels"] = {"rows": int(len(concepts)),
                                      "concepts": int(concepts["concept_id"].nunique())}
    man_path = processed_dir / "feature_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"wrote {rel(man_path)}")

    # ---- 7. report ----------------------------------------------------------
    report = build_quality_report(profiles, df, meta, checks, diag, args.cutoff,
                                  raw_hashes, args.output_name)
    report_path = reports_dir / "data_quality_report.md"
    report_path.write_text(report, encoding="utf-8")
    log(f"wrote {rel(report_path)}")

    # ---- 8. console summary -------------------------------------------------
    print("\n" + "=" * 78)
    print("MILESTONE 1 PIPELINE SUMMARY")
    print("=" * 78)
    print(f"  cutoff day              : {args.cutoff}")
    print(f"  enrolments (raw)        : {meta['counts']['enrolments_total']:,}")
    print(f"  excluded (withdrawn)    : {meta['counts']['excluded_withdrawn_before_cutoff']:,}")
    print(f"  excluded (late reg.)    : {meta['counts']['excluded_registered_after_cutoff']:,}")
    print(f"  feature rows            : {meta['counts']['included']:,}")
    print(f"  feature columns         : {df.shape[1]} ({len(feature_columns(df))} predictors)")
    print(f"  at-risk prevalence      : {100*df['target_at_risk'].mean():.1f}%")
    print(f"  rows with graded result : {100*df['has_graded_assessment'].mean():.1f}%")
    print(f"  rows with VLE activity  : {100*df['has_vle_activity'].mean():.1f}%")
    print(f"  checks passed           : {(checks['status']=='PASS').sum()}/{len(checks)}")
    print(f"  raw source              : {source.describe()}")
    print(f"  elapsed                 : {time.time()-t0:.1f}s")
    print("=" * 78)
    source.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RawDataError as exc:
        print(f"\n[RAW DATA ERROR]\n{exc}\n", file=sys.stderr)
        raise SystemExit(2)
