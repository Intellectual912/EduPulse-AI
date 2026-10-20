# EduPulse AI

**Early-Warning Performance Risk Classifier & Socratic Coaching Platform**

A college mini-project that will predict which students are heading for academic difficulty
*while there is still time to intervene*, and then coach them with an AI tutor grounded in their
own course material.

> **Status — Milestone 1 complete (20 October 2026).**
> Cleaned academic records and engineered a leakage-free tabular feature set from the OULAD dataset.
> No model has been trained, no LLM/RAG component exists yet, and there is no dashboard: those are
> Milestones 2–4. See [Roadmap](#roadmap).

---

## Table of contents

1. [What this milestone delivers](#what-this-milestone-delivers)
2. [Quick start](#quick-start)
3. [Dataset](#dataset)
4. [Project structure](#project-structure)
5. [The prediction design (the important part)](#the-prediction-design)
6. [Feature catalogue](#feature-catalogue)
7. [Outputs](#outputs)
8. [Command reference](#command-reference)
9. [Testing and verification](#testing-and-verification)
10. [Key findings about the data](#key-findings-about-the-data)
11. [Assumptions and limitations](#assumptions-and-limitations)
12. [Roadmap](#roadmap)
13. [Ethics, privacy and licensing](#ethics-privacy-and-licensing)
14. [References](#references)

---

## What this milestone delivers

| Requirement (Milestone 1, 20 Oct 2026) | Where it is satisfied |
|---|---|
| Clean academic performance records | `src/preprocess.py` — validation, missing-value policy, duplicate handling, exclusions ledger |
| Prepare tabular features with Pandas / Scikit-learn | `data/processed/student_features.csv` (27,522 × 70) |
| Exploratory data analysis | `notebooks/data_exploration.ipynb` (executed, 7 figures) |
| Data-quality report with **actual** findings | `reports/data_quality_report.md` (generated from the real files) |
| Structured for Milestones 2–4 | cutoff parameter, feature manifest, concept-label hook, forbidden-predictor guard |

---

## Quick start

```bash
# 1. Environment (Python 3.11+; verified on 3.13.16)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Get the data (~47 MB; verifies the published MD5, keeps the archive packed)
python src/download_data.py

# 3. Build the feature table (~25 s) and the data-quality report
python src/preprocess.py

# 4. Verify
python -m pytest tests -q            # 19 tests

# 5. Explore (optional)
jupyter notebook notebooks/data_exploration.ipynb
```

Expected final line of step 3:

```
  feature rows            : 27,522
  feature columns         : 70 (63 predictors)
  at-risk prevalence      : 44.1%
  checks passed           : 7/7
```

---

## Dataset

**Open University Learning Analytics Dataset (OULAD)**, CC BY 4.0, anonymised at source.

| | |
|---|---|
| Source | <https://research.stem.open.ac.uk/ouanalyse/dataset/> |
| Persistent copies | Figshare DOI [10.6084/m9.figshare.5081998](https://doi.org/10.6084/m9.figshare.5081998) · UCI [dataset 349](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) |
| Archive | `anonymisedData.zip` (46.75 MB) — kept packed and read in place |
| MD5 | `7412686fd77cf0e0ee1e8c3e9b354308` (verified after download) |
| Contents | 22 module-presentations, 7 modules, 32,593 enrolments, 28,785 students, 10,655,280 VLE interaction rows |
| Licence | CC BY 4.0 — attribution required, redistribution permitted (see [Ethics](#ethics-privacy-and-licensing)) |

The official page's download button has changed URL more than once, so `src/download_data.py`
tries the official host first and falls back to the authors' Figshare record and UCI, verifying
the MD5 in every case. The seven CSVs are placed in `data/raw/` and **never modified** by any script.

**Manual download** (if you have no network access in your environment):

1. Open <https://research.stem.open.ac.uk/ouanalyse/dataset/> and click *Download dataset*, or use
   <https://figshare.com/articles/dataset/5081998>.
2. Save `anonymisedData.zip` into `data/raw/`.
3. Run `python src/download_data.py` — it verifies the MD5. The pipeline then reads the
   tables directly from the zip; add `--extract` only if you want the plain CSVs.

### Data storage and workspace size

The unpacked dataset is **488 MB, of which `studentVle.csv` alone is 454 MB.** To stay inside
typical workspace quotas, `src/download_data.py` keeps the verified **46.8 MB archive** and does
*not* unpack it. `src/preprocess.py` reads the seven tables **straight out of the zip**, so you get
identical features without the 444 MB of extracted CSV ever existing on disk.

Both layouts are supported and produce byte-identical output (verified: MD5
`42c1417385a3409842a5ed2275c7973c` from either source):

| Layout | On disk | When to use |
|---|---|---|
| **archived** (default) | `data/raw/anonymisedData.zip` | normal use — smallest footprint |
| **extracted** | 7 CSVs in `data/raw/` (~488 MB) | when another tool needs plain files |

```bash
python src/download_data.py --extract   # unpack (~488 MB)
python src/download_data.py --repack    # delete the CSVs, keep the archive (frees 464 MB)
python src/download_data.py --check-only # report which layout is present
```

**If your workspace is size-limited**, run `--repack` after any `--extract`, and never commit
`data/raw/` (it is already in `.gitignore`).

### Tables and their grain

| File | Rows | Grain (one row per …) | Role |
|---|---|---|---|
| `courses.csv` | 22 | module-presentation | course length in days |
| `assessments.csv` | 206 | assessment | type, due day, weight |
| `studentInfo.csv` | 32,593 | **enrolment** | demographics + `final_result` (the label source) |
| `studentRegistration.csv` | 32,593 | enrolment | registration / unregistration day |
| `studentAssessment.csv` | 173,912 | submission | submitted day, banked flag, score |
| `vle.csv` | 6,364 | VLE resource | activity type of each `id_site` |
| `studentVle.csv` | 10,655,280 | interaction record | day × resource × clicks |

---

## Project structure

```
EduPulse-AI/
├── data/
│   ├── raw/                          # OULAD archive (read-only, never modified)
│   │   └── anonymisedData.zip        # 46.8 MB verified; read in place, not unpacked
│   └── processed/
│       ├── student_features.csv      # ★ the Milestone 1 deliverable (27,522 × 70)
│       ├── feature_manifest.json     # machine-readable feature dictionary + provenance
│       ├── assessment_catalog.csv    # assessment metadata incl. type-normalised weight
│       ├── cutoff_diagnostics.csv    # cutoff trade-off table
│       └── excluded_records.csv      # every excluded row and the reason
├── src/
│   ├── preprocess.py                 # ★ the reproducible pipeline
│   └── download_data.py              # dataset acquisition + MD5 verification
├── notebooks/
│   └── data_exploration.ipynb        # executed EDA with figures
├── reports/
│   ├── data_quality_report.md        # ★ generated, all numbers computed
│   └── figures/                      # 7 figures exported by the notebook
├── tests/
│   └── test_pipeline.py              # 19 verification tests incl. a leakage test
├── requirements.txt
├── README.md
└── .gitignore
```

---

## The prediction design

### Four different "records" — do not confuse them

| Record type | Grain | Example |
|---|---|---|
| **Raw student assessment record** | one submission | *student 11391 submitted assessment 1752 on day 18 with score 78* |
| **Student-course record** | one enrolment | *student 11391 in AAA-2013J, final_result = Pass* |
| **Weekly student activity record** | one student × one week | *student 11391 clicked 42 times in week 2* |
| **Derived student feature record** | **one enrolment × one cutoff** | *student 11391 at day 28: 2 assessments due, 1 graded, mean score 78, 356 clicks, 14 active days* |

Only the last one is what `student_features.csv` contains.

### Grain of the output table

```
one row  =  (code_module, code_presentation, id_student, cutoff_day)
```

Verified unique (0 duplicates). `cutoff_day` is stored as a column even though a single run
produces one cutoff, so that multi-cutoff datasets can be stacked later without ambiguity.

### Observation window, cutoff and outcome period

| Element | Choice |
|---|---|
| **Unit predicted** | one student-course-presentation enrolment |
| **Observation window** | course days −∞ … **28** (pre-course VLE activity included; it is known before day 0) |
| **Prediction cutoff** | **day 28** (~10% into a 234–269-day presentation) |
| **Outcome period** | days 29 … course end (~90% of the course still ahead) |
| **Label** | `target_at_risk` = 1 if `final_result ∈ {Fail, Withdrawn}`, else 0 |
| **Label source** | OULAD `final_result` — **derived** into a binary risk category by this pipeline |

**Why day 28?** Measured, not guessed (`data/processed/cutoff_diagnostics.csv`):

| Cutoff | Presentations with an assessment due | Enrolments with ≥1 graded result | Mean graded assessments |
|---|---|---|---|
| 14 | 1 / 22 | 3.3 % | 0.03 |
| 21 | 11 / 22 | 42.3 % | 0.42 |
| **28** | **16 / 22** | **62.8 %** | **0.66** |
| 42 | 19 / 22 | 71.6 % | 0.88 |
| 56 | 19 / 22 | 72.3 % | 1.38 |

Day 28 is the earliest point at which most presentations have produced gradeable evidence while
~90% of the course remains ahead. Day 56 gives a stronger model but leaves only three quarters of
the course to intervene. `--cutoff` is a first-class parameter so Milestone 2 can train at 28 and 56
and compare on identical code.

### How leakage is prevented

1. **Assessment features** use only assessments that were *both* due on/before the cutoff *and*
   submitted on/before the cutoff.
2. **Withdrawal** (`date_unregistration ≤ cutoff`) is used **only to exclude already-ended
   enrolments**, never as a predictor. 5,055 such enrolments are excluded and logged.
3. **`final_result` and `target_at_risk` are never predictors.** `assert_no_leakage()` fails the run
   if a forbidden column appears among the model inputs.
4. **No learned transformation is fitted** in this milestone — no imputer, scaler or encoder touches
   the data. Those will be fitted **inside the training split** in Milestone 2.
5. **Executable proof:** `tests/test_pipeline.py::test_features_are_invariant_to_post_cutoff_data`
   rebuilds the dataset with every post-cutoff student record deleted (VLE clicks, submissions,
   withdrawal dates) and re-runs the whole pipeline. All 70 columns must be identical — and they are.
6. **Course-design metadata is used in full** (assessment schedule, due dates, weights, course
   length, VLE catalogue). This is published before day 0 and is legitimately known; only
   *student-generated* records are windowed. This is stated explicitly in the report.

---

## Feature catalogue

63 predictors in four families. `data/processed/feature_manifest.json` holds the authoritative
machine-readable list; `reports/data_quality_report.md` §10 documents every column.

| Family | n | Examples |
|---|---|---|
| **Static enrolment attributes** | 9 | `gender`, `region`, `highest_education`, `imd_band` (+ `imd_band_missing`), `age_band`, `disability`, `num_of_prev_attempts`, `studied_credits` |
| **Registration** | 5 | `registration_day`, `registration_days_before_start`, `registered_after_course_start`, `days_since_registration_at_cutoff`, `registration_date_missing` |
| **Assessment (windowed)** | 22 | `n_assessments_due`, `n_assessments_submitted`, `n_assessments_graded`, `n_assessments_unsubmitted`, `assessment_submission_rate`, `n_late_submissions`, `late_submission_rate`, `mean_submission_delay_days`, `max_submission_delay_days`, `days_since_last_submission`, `mean_score`, `min_score`, `max_score`, `std_score`, `weighted_mean_score`, `first_score`, `latest_score`, `score_change`, `has_assessment_due`, `has_graded_assessment`, `n_assessments_banked`, `assessment_graded_rate` |
| **VLE activity (windowed)** | 22 | `vle_clicks_total`, `vle_clicks_precourse`, `vle_active_days`, `vle_active_weeks`, `vle_distinct_sites`, `vle_clicks_per_active_day`, `vle_activity_group_count`, `vle_clicks_{forum,content,quiz,collab,wiki,home,other}`, `vle_forum_share`, `vle_clicks_last7d`, `vle_clicks_prev7d`, `vle_click_change_last7`, `vle_click_share_last7`, `vle_weekly_slope`, `vle_days_since_last_activity`, `has_vle_activity`, `vle_clicks_wk0…wk3` |

Non-predictors carried for traceability: `code_module`, `code_presentation`, `id_student`,
`cutoff_day`, `course_length_days`, `days_cutoff_to_course_end`, `final_result`, `target_at_risk`.

### Two definitions worth remembering

* **`sum_click` is a count of recorded platform interactions, not study hours.** No feature in this
  project is called or interpreted as "study time".
* **A missing score is not a zero.** Only 173 of 173,912 submission rows have no score; those stay
  `NaN`. Unsubmitted assessments are counted in `n_assessments_unsubmitted` rather than scored 0.

---

## Outputs

| File | Rows | Description |
|---|---|---|
| `data/processed/student_features.csv` | 27,522 × 70 | the feature table at the documented grain |
| `data/processed/feature_manifest.json` | — | grain, target definition, feature groups, per-column descriptions, raw-file SHA-256s, environment |
| `data/processed/assessment_catalog.csv` | 206 | assessment metadata + `weight_type_normalised` |
| `data/processed/cutoff_diagnostics.csv` | 5 | the cutoff trade-off table above |
| `data/processed/excluded_records.csv` | 5,071 | every excluded enrolment + reason + offending date |
| `reports/data_quality_report.md` | — | generated report: profiling, semantics, leakage rules, ledger, feature dictionary |
| `reports/figures/*.png` | 7 | figures exported by the executed notebook |

**Row accounting** (asserted by the pipeline, not just documented):

```
32,593 enrolments
 −5,055 withdrawn on/before day 28
    −16 registered after day 28
= 27,522 rows in student_features.csv
```

**Assessment-row ledger** (reconciles exactly):

```
173,912 submission rows
  −2,865 attached to an assessment with no due date (all Exam; submitted around day 230–285)
  −149,004 for assessments due after day 28
= 22,043 for assessments due by day 28
  −650 submitted after day 28
= 21,393 usable as cutoff-time evidence
```

---

## Command reference

| Command | Purpose |
|---|---|
| `python src/download_data.py` | download and verify MD5, keep the archive packed |
| `python src/download_data.py --extract` | unpack the seven CSVs (~488 MB on disk) |
| `python src/download_data.py --repack` | delete unpacked CSVs, keep the verified archive |
| `python src/download_data.py --check-only` | report which layout is present |
| `python src/preprocess.py` | build features at the default day-28 cutoff |
| `python src/preprocess.py --cutoff 42 --output-name student_features_c42.csv` | sensitivity run at another cutoff |
| `python src/preprocess.py --skip-diagnostics` | faster run without the cutoff table |
| `python -m pytest tests -q` | run the 19 verification tests |
| `jupyter nbconvert --execute --inplace notebooks/data_exploration.ipynb` | re-execute the EDA |

All paths are relative to the project root; `--raw-dir`, `--processed-dir` and `--reports-dir`
override them. Raw files are opened read-only and are never written to.

---

## Testing and verification

`python -m pytest tests -q` → **19 passed**.

| Test | What it proves |
|---|---|
| `test_all_seven_raw_files_present` | inputs exist and are non-empty |
| `test_required_artefacts_exist` | every deliverable was written |
| `test_feature_table_reopens_with_expected_shape` | the CSV reopens as 27,522 × 70 |
| `test_grain_is_unique` | `(module, presentation, student, cutoff)` has 0 duplicates |
| `test_expected_feature_groups_present` | all documented columns exist |
| `test_target_is_derived_correctly_and_separate` | label matches `final_result`; never a predictor |
| `test_no_forbidden_or_future_columns_are_predictors` | leakage guard holds |
| `test_features_respect_the_cutoff` | no negative "days since"; weekly buckets only cover complete weeks |
| `test_bounded_features_stay_in_range` (×5) | rates ∈ [0,1], scores ∈ [0,100] |
| `test_counts_are_consistent` | graded ≤ submitted ≤ due; no negative clicks |
| `test_row_accounting_reconciles` | 27,522 + 5,071 = 32,593 |
| `test_excluded_students_are_absent_from_features` | excluded rows never reappear |
| **`test_features_are_invariant_to_post_cutoff_data`** | **deleting the future changes nothing → no leakage** |
| `test_report_reflects_actual_run` | the report quotes real counts, not placeholders |
| `test_manifest_matches_table` | manifest rows/columns match the CSV; all predictors documented |

Independent audit performed during development (not part of the suite): 300 randomly sampled
enrolments had `n_assessments_due`, `n_assessments_submitted`, `n_assessments_graded`, `mean_score`,
`max_submission_delay_days`, `vle_clicks_total`, `vle_distinct_sites`, `vle_active_days` and
`vle_clicks_precourse` recomputed from the raw CSVs — **0 mismatches**. The VLE click total was also
reconciled to the row level: the gap between `vle_clicks_total` and the sum of the weekly buckets
equals exactly the clicks recorded on day 28 for included enrolments (162,699).

---

## Key findings about the data

These changed the implementation and are expanded in `reports/data_quality_report.md` §3.

1. **`assessments.weight` is within-type, not a share of the final mark.** Per presentation the
   weights sum to 100, 200 or 300; per (presentation, type) they sum to 0/5/25/75/95/100/200.
   The pipeline normalises weight *within* assessment type.
2. **11 assessments have no due date** (all Exams). They account for 2,865 submission rows, all
   submitted around day 230–285 — end-of-course exams beyond any early cutoff. No early signal is lost.
3. **`is_banked` marks carry-overs** from a previous attempt (1,909 rows overall). Their scores are
   valid results but not new work, so they are excluded from submission-delay statistics.
4. **`studentVle.csv` is not unique on its documented key**: 8,459,320 distinct
   (module, presentation, student, site, day) keys for 10,655,280 rows; 787,168 exact duplicate rows.
   `sum_click` is additive, so aggregation sums interactions rather than dropping records.
5. **Withdrawal is often early**: 3,097 enrolments unregister on or before day 0; 5,055 by day 28.
   Keeping them would make the task tautological, so they are excluded and logged.
6. **Two presentations (EEE, GGG) have no assessment due by day 28** → 4,989 rows have structurally
   absent assessment features, preserved as `NaN` with `has_assessment_due = 0`.
7. **1,677 rows have a graded assessment but no `weighted_mean_score`** because their only graded
   work carried weight 0 in OULAD (BBB-2014J). Documented, not a bug.
8. **OULAD has no concept labels.** No question-level items, no concept tags, no prerequisite graph —
   see [Assumptions and limitations](#assumptions-and-limitations).

---

## Assumptions and limitations

1. **Concept-level mastery tracking (faculty Feature 2) cannot be built from OULAD.**
   Results exist only per whole assessment (TMA/CMA/Exam). Inventing concept tags would fabricate
   labels, so this milestone instead prepares the join point: `assessment_catalog.csv` +
   `load_concept_labels()` in `src/preprocess.py`, which will validate and activate a future
   `assessment_concepts.csv` (`id_assessment, concept_id, weight_in_assessment`) without any change
   to the pipeline. Knowledge Tracing needs per-item response sequences, which OULAD does not
   contain at all.
2. **No demographics are used to *define* risk.** Region, IMD band and disability are retained as
   features because they are available at enrolment, but Milestone 2 must audit fairness across them;
   they must never justify an intervention on their own.
3. **The label is the *final* outcome of the presentation**, observed over the following ~240 days.
   It is valid for offline training but it is definitionally unavailable in a live deployment, where
   a proxy label (e.g. "failed the next assessment") will be needed.
4. **Clicks ≠ study time.** All VLE features are usage signals and are named accordingly.
5. **Presentation effects.** At-risk rates range from 24.7% (AAA-2013J) to 55.7% (CCC-2014B). Because
   `code_module` is an identifier rather than a predictor, Milestone 2 should evaluate on
   **held-out presentations**, not random rows, to avoid optimistic scores.
6. **Class imbalance is mild but real**: 44.1% at-risk after exclusions, so accuracy alone will be a
   misleading metric — precision, recall and AUC will be required.

---

## Roadmap

| Milestone | Date | Scope | Status |
|---|---|---|---|
| **1** | 20 Oct 2026 | Clean records, engineer tabular features; EDA + data-quality report | ✅ **complete** |
| **2** | 10 Nov 2026 | Train and validate the risk classifier (XGBoost / Random Forest, calibrated probabilities, fairness audit, presentation-wise validation) | next |
| **3** | 1 Dec 2026 | Personalised study guidance: LLM coach + RAG over the syllabus, micro-learning schedules, diagnostic questions | planned |
| **4** | 18 Dec 2026 | Student and faculty dashboards, public URL, evaluation-ready repository | planned |

**Immediate next steps for Milestone 2** (detailed at the end of the chat report):

1. Split by presentation (grouped `GroupKFold`) and fit imputers/encoders **inside** each training fold.
2. Baseline: logistic regression on the 63 features → then XGBoost; report AUC, precision/recall at a
   chosen alert budget, and calibration.
3. Handle the two structurally different cohorts (with/without assessment signal by day 28) —
   either two models or an explicit `has_assessment_due` interaction.
4. Re-run at `--cutoff 42` and `56` to quantify the accuracy/earliness trade-off for the report.
5. Fairness audit across `gender`, `age_band`, `disability`, `imd_band` before any deployment claim.

---

## Ethics, privacy and licensing

* OULAD is **anonymised at source** and released under **CC BY 4.0**; attribution is required.
* Even so, `.gitignore` excludes `data/raw/` and `data/processed/` by default. Committing
  student-level records should only happen after confirming it is permitted for the submission; the
  README and `src/download_data.py` let a reviewer regenerate the data instead.
* **No credentials belong in this repository.** `.env*`, `*.key`, `*.pem` and `secrets.json` are
  ignored, and no API key is used or required by Milestone 1.
* A risk score is a **decision-support signal, not a verdict**. Any future dashboard must present it
  with uncertainty, let students see and contest their own data, and never expose one student's
  records to another.

## References

* Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). *Open University Learning Analytics dataset.*
  Scientific Data 4, 170171. <https://doi.org/10.1038/sdata.2017.171>
* OU Analyse dataset page: <https://research.stem.open.ac.uk/ouanalyse/dataset/>
* UCI ML Repository, OULAD: <https://archive.ics.uci.edu/dataset/349>
* XGBoost documentation: <https://xgboost.readthedocs.io/en/stable/get_started.html>
* DeepLearning.AI (background for Milestones 2–3): <https://www.deeplearning.ai/>
