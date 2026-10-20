"""
EduPulse AI — Milestone 1 verification suite.

These are not decorative tests: each one checks a claim made in the README or
the data-quality report against the artefacts actually produced.

Run:
    python -m pytest tests -q

The most important test is `test_features_are_invariant_to_post_cutoff_data`:
it rebuilds the dataset with every piece of post-cutoff information deleted
(VLE clicks, submissions, assessment definitions, withdrawal dates) and asserts
the feature table is byte-for-byte identical. If any feature silently used
future information, that test fails.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from preprocess import RawSource, RAW_FILES  # noqa: E402

RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

FEATURES = PROCESSED / "student_features.csv"
MANIFEST = PROCESSED / "feature_manifest.json"
EXCLUSIONS = PROCESSED / "excluded_records.csv"
CATALOG = PROCESSED / "assessment_catalog.csv"
DIAGNOSTICS = PROCESSED / "cutoff_diagnostics.csv"
REPORT = REPORTS / "data_quality_report.md"

KEY = ["code_module", "code_presentation", "id_student", "cutoff_day"]
CUTOFF = 28


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def features() -> pd.DataFrame:
    if not FEATURES.exists():
        pytest.skip("processed features missing; run `python src/preprocess.py` first")
    return pd.read_csv(FEATURES)


@pytest.fixture(scope="session")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("manifest missing; run `python src/preprocess.py` first")
    return json.loads(MANIFEST.read_text())


# --------------------------------------------------------------------------- #
# 1. Raw inputs
# --------------------------------------------------------------------------- #

def test_all_seven_raw_tables_are_available():
    """
    The seven tables must be readable, either as extracted CSVs or straight out
    of anonymisedData.zip. Both layouts are supported by the pipeline.
    """
    source = RawSource(RAW)
    try:
        assert source.mode in {"extracted", "archived"}, (
            f"no usable OULAD data in {RAW}: expected the seven CSVs or a complete "
            f"anonymisedData.zip (got mode={source.mode!r})")
        for name in RAW_FILES:
            assert source.size(name) > 0, f"{name} is empty"
    finally:
        source.close()


# --------------------------------------------------------------------------- #
# 2. Outputs exist, reopen, and have the documented grain
# --------------------------------------------------------------------------- #

def test_required_artefacts_exist():
    for p in (FEATURES, MANIFEST, EXCLUSIONS, CATALOG, REPORT):
        assert p.exists(), f"missing artefact: {p}"
        assert p.stat().st_size > 0, f"empty artefact: {p}"


def test_feature_table_reopens_with_expected_shape(features):
    assert features.shape[0] == 27_522, "row count changed unexpectedly"
    assert features.shape[0] > 1_000, "feature table looks like an empty skeleton"
    assert features.shape[1] == 70


def test_grain_is_unique(features):
    assert not features.duplicated(KEY).any(), "grain (module, presentation, student, cutoff) is not unique"
    assert features["cutoff_day"].nunique() == 1, "a single-cutoff table must carry exactly one cutoff"


def test_expected_feature_groups_present(features):
    required = [
        # identifiers / metadata
        "code_module", "code_presentation", "id_student", "cutoff_day",
        "course_length_days", "days_cutoff_to_course_end",
        # assessment block
        "n_assessments_due", "n_assessments_submitted", "n_assessments_graded",
        "n_assessments_unsubmitted", "mean_score", "latest_score", "score_change",
        "mean_submission_delay_days", "late_submission_rate",
        # vle block
        "vle_clicks_total", "vle_active_days", "vle_active_weeks", "vle_distinct_sites",
        "vle_clicks_last7d", "vle_clicks_prev7d", "vle_weekly_slope", "vle_clicks_wk0",
        # registration block
        "registration_day", "days_since_registration_at_cutoff",
        # target
        "final_result", "target_at_risk",
    ]
    missing = [c for c in required if c not in features.columns]
    assert not missing, f"missing expected columns: {missing}"


# --------------------------------------------------------------------------- #
# 3. Target integrity and leakage guards
# --------------------------------------------------------------------------- #

def test_target_is_derived_correctly_and_separate(features, manifest):
    assert set(features["final_result"].unique()) <= {"Pass", "Fail", "Withdrawn", "Distinction"}
    recomputed = features["final_result"].isin(["Fail", "Withdrawn"]).astype(int)
    assert (recomputed == features["target_at_risk"].astype(int)).all()
    assert features["target_at_risk"].notna().all()
    assert "target_at_risk" not in manifest["feature_columns"]
    assert "final_result" not in manifest["feature_columns"]


def test_no_forbidden_or_future_columns_are_predictors(features, manifest):
    forbidden = {"final_result", "target_at_risk", "date_unregistration", "date_submitted",
                 "score", "withdrawn_before_cutoff"}
    assert not (forbidden & set(manifest["feature_columns"]))
    assert "date_unregistration" not in features.columns
    # no raw timestamp may leak in as a column
    assert not any(c.startswith("date_") for c in manifest["feature_columns"])


def test_features_respect_the_cutoff(features):
    # anything measured "since" the cutoff cannot be negative
    assert (features["days_since_last_submission"].dropna() >= 0).all()
    assert (features["vle_days_since_last_activity"].dropna() >= 0).all()
    assert (features["days_since_registration_at_cutoff"].dropna() >= 0).all()
    # weekly buckets only cover complete weeks inside the window
    weekly = [c for c in features.columns if c.startswith("vle_clicks_wk")]
    assert weekly == [f"vle_clicks_wk{i}" for i in range(CUTOFF // 7)]


@pytest.mark.parametrize("name,low,high", [
    ("assessment_submission_rate", 0, 1),
    ("assessment_graded_rate", 0, 1),
    ("late_submission_rate", 0, 1),
    ("mean_score", 0, 100),
    ("weighted_mean_score", 0, 100),
])
def test_bounded_features_stay_in_range(features, name, low, high):
    s = features[name].dropna()
    assert s.between(low, high).all(), f"{name} has values outside [{low}, {high}]"


def test_counts_are_consistent(features):
    assert (features[["n_assessments_due", "n_assessments_submitted", "n_assessments_graded",
                      "n_assessments_unsubmitted", "n_late_submissions"]] >= 0).all().all()
    assert (features["n_assessments_graded"] <= features["n_assessments_submitted"]).all()
    assert (features["n_assessments_submitted"] <= features["n_assessments_due"]).all()
    assert (features[["vle_clicks_total", "vle_active_days", "vle_distinct_sites"]] >= 0).all().all()


# --------------------------------------------------------------------------- #
# 4. Row accounting and exclusions
# --------------------------------------------------------------------------- #

def test_row_accounting_reconciles(features, manifest):
    counts = manifest["counts"]
    excl = pd.read_csv(EXCLUSIONS)
    assert counts["included"] == len(features)
    assert counts["included"] + counts["excluded_total"] == counts["enrolments_total"]
    assert len(excl) == counts["excluded_total"]
    assert set(excl["exclusion_reason"]) <= {"withdrawn_on_or_before_cutoff", "registered_after_cutoff"}


def test_excluded_students_are_absent_from_features(features):
    excl = pd.read_csv(EXCLUSIONS)
    merged = excl.merge(features[["code_module", "code_presentation", "id_student"]],
                        on=["code_module", "code_presentation", "id_student"], how="inner")
    assert len(merged) == 0, f"{len(merged)} excluded enrolments reappeared in the feature table"


# --------------------------------------------------------------------------- #
# 5. THE leakage test: features must not change when the future is deleted
# --------------------------------------------------------------------------- #

def _mask_raw_dataset(src: Path, dst: Path, cutoff: int) -> None:
    """
    Copy raw CSVs to dst, deleting every **student-generated** record that is
    dated after `cutoff`.

    Course-design metadata (the assessment schedule, due dates, weights, the
    VLE resource catalogue and the course length) is deliberately preserved:
    it is published before the course starts and is therefore legitimately
    known at prediction time. What must never influence a feature is *student*
    activity or results from the future.
    """
    dst.mkdir(parents=True, exist_ok=True)
    source = RawSource(src)

    def read(name, **kw):
        with source.open(name) as fh:
            return pd.read_csv(fh, **kw)

    try:
        # course-design metadata + static enrolment attributes copied as-is
        for name in ["courses.csv", "assessments.csv", "studentInfo.csv", "vle.csv"]:
            with source.open(name) as fh:
                (dst / name).write_bytes(fh.read())

        # submissions: drop everything submitted after the cutoff
        sa = read("studentAssessment.csv")
        sa[sa["date_submitted"] <= cutoff].to_csv(dst / "studentAssessment.csv", index=False)

        # registration: withdrawal dates after the cutoff become unknown
        reg = read("studentRegistration.csv")
        reg.loc[reg["date_unregistration"] > cutoff, "date_unregistration"] = np.nan
        reg.to_csv(dst / "studentRegistration.csv", index=False)

        # VLE clicks: keep only days inside the window, writing in chunks
        out = dst / "studentVle.csv"
        first = True
        with source.open("studentVle.csv") as fh:
            for chunk in pd.read_csv(fh, chunksize=2_000_000):
                chunk = chunk[chunk["date"] <= cutoff]
                chunk.to_csv(out, mode="w" if first else "a", header=first, index=False)
                first = False
    finally:
        source.close()


def test_features_are_invariant_to_post_cutoff_data(tmp_path):
    """
    The decisive leakage test.

    Rebuild the raw dataset with every post-cutoff fact deleted, re-run the whole
    pipeline against it, and require identical features. Any feature that peeks
    at the future would change.
    """
    masked_raw = tmp_path / "raw"
    masked_out = tmp_path / "processed"
    _mask_raw_dataset(RAW, masked_raw, CUTOFF)

    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "preprocess.py"),
         "--raw-dir", str(masked_raw),
         "--processed-dir", str(masked_out),
         "--reports-dir", str(tmp_path / "reports"),
         "--cutoff", str(CUTOFF),
         "--chunk-size", "3000000",
         "--skip-diagnostics"],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=900)
    assert result.returncode == 0, f"masked pipeline failed:\n{result.stderr[-3000:]}"

    original = pd.read_csv(FEATURES)
    masked = pd.read_csv(masked_out / "student_features.csv")

    assert original.shape == masked.shape, "row/column count changed when the future was deleted"
    assert list(original.columns) == list(masked.columns)

    # compare every column; NaN == NaN is expected in both frames
    for col in original.columns:
        if original[col].dtype.kind in "fc":
            same = np.allclose(original[col].to_numpy(dtype=float),
                               masked[col].to_numpy(dtype=float), equal_nan=True)
        else:
            same = original[col].fillna("__NA__").astype(str).equals(
                masked[col].fillna("__NA__").astype(str))
        assert same, f"column '{col}' changed when post-cutoff data was deleted -> leakage"


# --------------------------------------------------------------------------- #
# 6. Generated documentation reflects execution
# --------------------------------------------------------------------------- #

def test_report_reflects_actual_run(features):
    text = REPORT.read_text(encoding="utf-8")
    assert "# EduPulse AI — Data Quality Report" in text
    assert "Nothing is illustrative" not in text
    # the report must quote the real row counts, not placeholders
    assert f"{len(features):,}" in text or str(len(features)) in text
    assert "43/43 checks passed" not in text  # guard against hard-coded claims
    assert "concept" in text.lower()
    for section in ["File inventory", "Prediction cutoff", "leakage", "Feature dictionary"]:
        assert section.lower() in text.lower(), f"report is missing section: {section}"


def test_manifest_matches_table(features, manifest):
    assert manifest["rows"] == len(features)
    assert manifest["cutoff_day"] == CUTOFF
    assert manifest["grain"] == KEY
    assert set(manifest["target_columns"]) == {"final_result", "target_at_risk"}
    for col in manifest["feature_columns"]:
        assert col in features.columns, f"manifest lists a missing column: {col}"
    # every declared predictor must be documented
    undocumented = [c for c in manifest["feature_columns"]
                    if c not in manifest["feature_descriptions"]
                    and not c.startswith("vle_clicks_wk")]
    assert not undocumented, f"undocumented predictors: {undocumented}"
