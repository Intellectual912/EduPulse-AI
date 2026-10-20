# EduPulse AI — Data Quality Report (Milestone 1)

_Generated automatically by `src/preprocess.py` on 2026-10-20 09:59 UTC from the actual raw files. Cutoff = course day 28._

Every number below was computed from the files in `data/raw/`; nothing is illustrative.

## 1. File inventory and checksums

| file | rows | columns | sha256 |
|---|---|---|---|
| courses.csv | 22 | 3 | 4f16eee7454b15e1… |
| assessments.csv | 206 | 6 | 8cc738fb88ad7605… |
| studentInfo.csv | 32,593 | 12 | 7e6f3e474a5eee00… |
| studentRegistration.csv | 32,593 | 5 | 0d32676285372aaf… |
| studentAssessment.csv | 173,912 | 5 | fd5320786328d05a… |
| vle.csv | 6,364 | 6 | d1b28303dea802ad… |
| studentVle.csv | 10,655,280 | 6 | 52668253d876c5be… |


**Raw data source for this run:** archive anonymisedData.zip (read without extracting). The pipeline reads the seven tables either from extracted CSVs or directly from `anonymisedData.zip`; both layouts produce identical features.


Source: OULAD (Kuzilek, Hlosta & Zdrahal, 2017), CC BY 4.0, retrieved from the authors' Figshare record (DOI 10.6084/m9.figshare.5081998); archive MD5 `7412686fd77cf0e0ee1e8c3e9b354308` verified after download.

## 2. Per-file profile

### `courses.csv` — 22 rows × 3 columns

- **Relationship:** One row per course presentation; supplies module_presentation_length.

**Data types**

| column | dtype |
|---|---|
| code_module | object |
| code_presentation | object |
| module_presentation_length | int64 |


**Missing values:** none.


**Duplicate rows:** 0



**Key candidates** (0 duplicates = usable as a join key)

| key candidate | duplicate rows |
|---|---|
| (module, presentation) | 0 |


**Unexpected / invalid values:** none detected by the domain checks.


**Low-cardinality columns:** `code_module` (7), `code_presentation` (4), `module_presentation_length` (7)



### `assessments.csv` — 206 rows × 6 columns

- **Relationship:** Assessment metadata; joins to studentAssessment on id_assessment (1-to-many).

**Data types**

| column | dtype |
|---|---|
| code_module | object |
| code_presentation | object |
| id_assessment | int64 |
| assessment_type | object |
| date | float64 |
| weight | float64 |


**Missing values**

| column | missing | pct |
|---|---|---|
| date | 11 | 5.34% |


**Duplicate rows:** 0



**Key candidates** (0 duplicates = usable as a join key)

| key candidate | duplicate rows |
|---|---|
| id_assessment | 0 |
| (module, presentation, id_assessment) | 0 |


**Unexpected / invalid values:** none detected by the domain checks.


**Low-cardinality columns:** `code_module` (7), `code_presentation` (4), `assessment_type` (3)



### `studentInfo.csv` — 32,593 rows × 12 columns

- **Relationship:** One row per enrolment; the spine of the feature table. Holds final_result.

**Data types**

| column | dtype |
|---|---|
| code_module | object |
| code_presentation | object |
| id_student | int64 |
| gender | object |
| region | object |
| highest_education | object |
| imd_band | object |
| age_band | object |
| num_of_prev_attempts | int64 |
| studied_credits | int64 |
| disability | object |
| final_result | object |


**Missing values**

| column | missing | pct |
|---|---|---|
| imd_band | 1,111 | 3.41% |


**Duplicate rows:** 0



**Key candidates** (0 duplicates = usable as a join key)

| key candidate | duplicate rows |
|---|---|
| (module, presentation, student) | 0 |


**Unexpected / invalid values:** none detected by the domain checks.


**Low-cardinality columns:** `code_module` (7), `code_presentation` (4), `gender` (2), `region` (13), `highest_education` (5), `imd_band` (10), `age_band` (3), `num_of_prev_attempts` (7), `disability` (2), `final_result` (4)



### `studentRegistration.csv` — 32,593 rows × 5 columns

- **Relationship:** One row per enrolment; registration and withdrawal dates.

**Data types**

| column | dtype |
|---|---|
| code_module | object |
| code_presentation | object |
| id_student | int64 |
| date_registration | float64 |
| date_unregistration | float64 |


**Missing values**

| column | missing | pct |
|---|---|---|
| date_registration | 45 | 0.14% |
| date_unregistration | 22,521 | 69.1% |


**Duplicate rows:** 0



**Key candidates** (0 duplicates = usable as a join key)

| key candidate | duplicate rows |
|---|---|
| (module, presentation, student) | 0 |


**Unexpected / invalid values:** none detected by the domain checks.


**Low-cardinality columns:** `code_module` (7), `code_presentation` (4)



### `studentAssessment.csv` — 173,912 rows × 5 columns

- **Relationship:** Submission rows; joins to assessments on id_assessment.

**Data types**

| column | dtype |
|---|---|
| id_assessment | int64 |
| id_student | int64 |
| date_submitted | int64 |
| is_banked | int64 |
| score | float64 |


**Missing values**

| column | missing | pct |
|---|---|---|
| score | 173 | 0.1% |


**Duplicate rows:** 0



**Key candidates** (0 duplicates = usable as a join key)

| key candidate | duplicate rows |
|---|---|
| (id_assessment, id_student) | 0 |


**Unexpected / invalid values:** none detected by the domain checks.


**Low-cardinality columns:** `is_banked` (2)



### `vle.csv` — 6,364 rows × 6 columns

- **Relationship:** VLE resource metadata; joins to studentVle on id_site (1-to-many).

**Data types**

| column | dtype |
|---|---|
| id_site | int64 |
| code_module | object |
| code_presentation | object |
| activity_type | object |
| week_from | float64 |
| week_to | float64 |


**Missing values**

| column | missing | pct |
|---|---|---|
| week_from | 5,243 | 82.39% |
| week_to | 5,243 | 82.39% |


**Duplicate rows:** 0



**Key candidates** (0 duplicates = usable as a join key)

| key candidate | duplicate rows |
|---|---|
| id_site | 0 |


**Unexpected / invalid values:** none detected by the domain checks.


**Low-cardinality columns:** `code_module` (7), `code_presentation` (4), `activity_type` (20)



### `studentVle.csv` — 10,655,280 rows × 6 columns

- **Relationship:** One row per (module, presentation, student, VLE resource, day) *intended*; in practice the key is not unique (see duplicates). sum_click is an additive interaction count.

**Data types**

| column | dtype |
|---|---|
| code_module | int32/int16/category |
| code_presentation | int32/int16/category |
| id_student | int32/int16/category |
| id_site | int32/int16/category |
| date | int32/int16/category |
| sum_click | int32/int16/category |


**Missing values:** none.


**Duplicate rows:** 787,170
 (7.39% of the file)



**Key structure:** 8,459,320 distinct (module, presentation, student, site, date) keys for 10,655,280 rows → 2,195,960 rows share a key with another row, of which 890,496 keys carry *different* `sum_click` values. The key is therefore **not unique**, and `sum_click` behaves as an additive interaction count. Multiplicity: 1 rows → 6,844,815 keys, 2 rows → 1,311,841 keys, 3-5 rows → 250,886 keys, 6-10 rows → 51,778 keys, >10 rows → 0 keys.

**Value ranges:** date -25…269; negative `sum_click` = 0; zero `sum_click` = 0; 26,074 distinct students, 6,268 distinct sites, 22 module-presentations.

**Referential integrity:** 0 sites missing from `vle.csv`; 0 student ids missing from `studentInfo.csv`; 3,365 enrolments have no VLE row at all.

**Unexpected / invalid values:** none detected by the domain checks.



## 3. Semantic findings that changed the design

1. **`assessments.weight` is within-type, not a share of the final mark.** Summing `weight` per presentation gives 100–300 (not 100). Summed per (presentation, assessment_type) the totals are: 0→10 presentations, 5→3 presentations, 25→3 presentations, 75→3 presentations, 95→3 presentations, 100→33 presentations, 200→2 presentations. A naive weighted average over raw `weight` would therefore be wrong; this pipeline normalises `weight` **within** assessment type. Note that some assessments carry weight 0 (e.g. a formative TMA), so they contribute nothing to `weighted_mean_score` by design.

2. **11 assessments have no due date** (Exam × 11). They account for 2,865 submission rows, and because their timing is unknown they cannot enter a windowed feature. This costs no early-warning signal: those submissions occur on days 230–285 (median 243), matching the dated end-of-course exams (days 222–261) — they are end-of-course exams, i.e. far beyond any early cutoff, not early evidence.

3. **`is_banked` marks carry-overs from a previous attempt** (508 banked submissions inside the window). Their scores are valid results, but they are not new work, so they are excluded from submission-delay statistics.

4. **A score of 0 is not the same as a missing score.** Only 173 of 173,912 submission rows lack a score; these are left as NaN and are never replaced by 0. Unsubmitted assessments are counted in `n_assessments_unsubmitted`, not scored as zero.

5. **Withdrawal is often early:** 3,097 enrolments carry an unregistration date on or before day 0, and 5,055 end on or before the day-28 cutoff. Those rows are excluded from the cohort because their outcome is already known at prediction time; keeping them would be tautological.

6. **VLE rows are not unique per (student, site, day)** and clicks are *not* study hours. All VLE features are documented as recorded platform interactions.

7. **OULAD contains no concept or question-level labels.** Assessments are whole TMA/CMA/Exam units; there is no item bank, no concept tag and no prerequisite graph. Concept-level mastery tracking (faculty Feature 2) therefore cannot be implemented from this dataset as-is — see §8.

## 4. Prediction cutoff: the trade-off

The cutoff is the last course day included in features. Earlier = more useful warning, but fewer assessment results exist. This table is computed from the raw data:

| cutoff_day | cohort_after_exclusions | presentations_with_assessment | presentations_total | pct_enrolments_with_ge1_graded | mean_graded_assessments | vle_rows_in_window_pct | vle_clicks_in_window_pct |
|---|---|---|---|---|---|---|---|
| 14 | 28061 | 1 | 22 | 3.3 | 0.03 | 17.1 | 11.7 |
| 21 | 27883 | 11 | 22 | 42.3 | 0.42 | 22.5 | 16 |
| 28 | 27522 | 16 | 22 | 62.8 | 0.66 | 26.7 | 19.1 |
| 42 | 27012 | 19 | 22 | 71.6 | 0.88 | 33.8 | 24.2 |
| 56 | 26513 | 19 | 22 | 72.3 | 1.38 | 40.7 | 29.1 |


**Chosen cutoff: day 28** (about 28/270 ≈ 10% of a presentation). At this point the course has ~90% of its length still ahead, which is genuinely early-warning, and most enrolments already have VLE signal. Consequences are documented rather than hidden:

- Some presentations (EEE, GGG) have **no assessment due by day 28**, so their assessment features are structurally absent (NaN) and `has_assessment_due = 0`; this is preserved explicitly instead of being filled with zeros.

- `--cutoff` is a first-class parameter; the November model can re-run at 42/56 days and compare, because every downstream artefact records `cutoff_day`.

## 5. Prediction target and leakage control

- **Unit predicted:** one student-course-presentation enrolment (grain of `student_features.csv`).
- **Observation window:** course days −∞ … 28 (pre-course access included; it is known before the cutoff).
- **Prediction cutoff:** day 28.
- **Outcome period:** days 29 … course end (234–269 days, from `courses.csv`).
- **Label:** `target_at_risk = 1` if `final_result ∈ {Fail, Withdrawn}`, else 0. `final_result` is supplied by OULAD; the binary risk category is **derived here**.
- **Prevalence in the processed cohort:** 44.1% at-risk (12,143 of 27,522).

Rules enforced in code (see `assert_no_leakage`):

| Rule | Enforcement |
|---|---|
| `final_result` may not be a predictor | Outcome label recorded after the course ends. |
| `target_at_risk` may not be a predictor | Derived outcome label. |
| `date_unregistration` may not be a predictor | Withdrawal date; post-cutoff for most students. |
| `withdrawn_before_cutoff` may not be a predictor | Defines cohort exclusion, not a predictor. |
| `date_submitted` may not be a predictor | Raw timestamp; only windowed derivations may be used. |
| `score` may not be a predictor | Raw per-assessment score; only windowed aggregates may be used. |
| `module_presentation_length` may not be a predictor | Course metadata used for the outcome period description. |
| Assessment results only if due **and** submitted by the cutoff | window filter in `build_assessment_features` |
| No withdrawal after the cutoff as a feature | only `date_unregistration <= cutoff` is used, and only to exclude rows |
| Course-design metadata may be used in full | the assessment schedule, due dates, weights, course length and VLE resource catalogue are published before day 0; only *student-generated* records are windowed. `weighted_mean_score` therefore depends on the whole schedule, which is intended and is covered by the leakage test |
| No fitting of imputers/scalers on the full dataset | this milestone only *derives* features; no learned transformation is applied. Scaling/imputation will be fitted inside the training split in Milestone 2 |

## 6. Row accounting and exclusions

| stage | rows |
|---|---|
| Enrolments in studentInfo.csv | 32,593 |
| − withdrawn on/before cutoff | −5,055 |
| − registered after cutoff | −16 |
| = rows in processed feature table | 27,522 |


Every excluded row is written to `data/processed/excluded_records.csv` with its reason and the offending date value; nothing is dropped silently.


**Assessment-row ledger** (reconciles exactly: yes)

| stage | rows | why |
|---|---|---|
| studentAssessment rows read | 173,912 | raw table |
| − assessment has no due date | −2,865 | timing unknown; end-of-course exams (see §3.2) |
| − assessment due after the cutoff | −149,004 | future information at prediction time |
| = rows for assessments due by the cutoff | 22,043 |  |
| − submitted after the cutoff | −650 | the result did not exist yet at the cutoff |
| = submissions usable as cutoff-time evidence | 21,393 |  |


## 7. Processed feature table

`student_features.csv` — 27,522 rows × 70 columns; 63 usable predictors.

**Grain:** one row per `(code_module, code_presentation, id_student, cutoff_day)` — unique? **True**.

**Missingness in the feature table (top columns)**

| feature | missing_pct |
|---|---|
| std_score | 96.7 |
| score_change | 96.7 |
| weighted_mean_score | 33.2 |
| max_submission_delay_days | 28.8 |
| late_submission_rate | 28.8 |
| mean_submission_delay_days | 28.8 |
| mean_score | 27.1 |
| days_since_last_submission | 27.1 |
| first_score | 27.1 |
| min_score | 27.1 |
| max_score | 27.1 |
| latest_score | 27.1 |
| assessment_submission_rate | 18.1 |
| assessment_graded_rate | 18.1 |
| imd_band | 3.7 |
| vle_forum_share | 3.3 |
| vle_click_share_last7 | 3.3 |
| vle_days_since_last_activity | 3.3 |
| vle_clicks_per_active_day | 3.3 |


Structural NaNs are expected: `mean_score` is NaN when no assessment was graded by the cutoff, and ratio features are NaN when their denominator is zero. These are *unknown*, not zero.


`weighted_mean_score` is additionally absent for 1,677 rows whose only graded work by the cutoff carried **zero weight** in OULAD (a formative assessment). Those rows are listed here; they are not a processing error:


| code_module | code_presentation | rows |
|---|---|---|
| BBB | 2014J | 1677 |


## 8. Concept-level mastery (faculty Feature 2): what OULAD does and does not support

OULAD stores results per **whole assessment** (`id_assessment` → TMA/CMA/Exam). It has no question-level items, no concept tags, and no prerequisite structure, so concept mastery and prerequisite-gap detection **cannot** be computed from this dataset without inventing labels. What this milestone prepares instead:

1. `data/processed/assessment_catalog.csv` — one row per assessment (module, presentation, type, due day, raw weight, type-normalised weight). A future `assessment_concepts.csv` (`id_assessment, concept_id, weight_in_assessment`) joins here at a 1-to-many grain.
2. `src/preprocess.py::load_concept_labels()` — optional, validated hook that activates concept tagging the moment such a file is provided, with no change to the pipeline.
3. The VLE activity-type groups (forum/content/quiz/collab/wiki) give coarse *behavioural* cues; they are not concepts and are not presented as such.
4. Knowledge Tracing (BKT/DKT) additionally requires per-item response sequences, which OULAD does not contain; it needs a different data source.


## 9. Executed validation checks

| check | status | detail |
|---|---|---|
| grain_uniqueness | PASS | duplicate rows on (module, presentation, student, cutoff): 0 |
| single_cutoff | PASS | cutoff_day values: [28] |
| row_accounting | PASS | 27522 included + 5071 excluded = 32593 vs 32593 enrolments |
| no_target_nulls | PASS | null targets: 0 |
| identifiers_present | PASS | key columns present: True |
| assessment_window | PASS | n_assessments_due is non-negative for all rows |
| no_future_withdrawal_feature | PASS | withdrawal date is not present in the feature table |


**Result:** 7/7 checks passed.


## 10. Feature dictionary

| column | group | description | dtype |
|---|---|---|---|
| code_module | identifier | OULAD module code (7 modules: AAA-GGG). Identifier, not a predictor. | object |
| code_presentation | identifier | Presentation code: 'B' = February start, 'J' = October start. Identifier, not a predictor. | object |
| id_student | identifier | Anonymised student id. Identifier used for grouping and joins; never a predictor. | int64 |
| cutoff_day | identifier | Course-relative day at which observation stops (day 0 = course start). | int16 |
| course_length_days | metadata | Presentation length in days from courses.csv; used to define the outcome period. | int16 |
| days_cutoff_to_course_end | metadata | course_length_days - cutoff_day: how much of the course is still in the future when the prediction is made. | int16 |
| gender | static | Gender as declared at enrolment (M/F). Available before the cutoff. | object |
| region | static | UK/other region of the student at enrolment. | object |
| highest_education | static | Highest education level declared at enrolment. | object |
| imd_band | static | Index of Multiple Deprivation band; missing for 3.4% of enrolments and left as NaN (never zero-filled). | object |
| imd_band_missing | static | 1 if imd_band was absent at enrolment, else 0 (explicit missingness indicator). | int8 |
| age_band | static | Age band at enrolment. | object |
| disability | static | Declared disability flag (Y/N). | object |
| num_of_prev_attempts | static | Number of previous attempts at this module, at enrolment. | int64 |
| studied_credits | static | Credits the student is studying in this presentation, at enrolment. | int64 |
| registration_day | registration | Registration day relative to course start (negative = before start). NaN when absent in OULAD (45 enrolments). | float64 |
| registration_days_before_start | registration | -registration_day; positive means registered early. NaN when registration day is missing. | float64 |
| registered_after_course_start | registration | 1 if the student registered on/after day 0 of the presentation. | int8 |
| days_since_registration_at_cutoff | registration | cutoff_day - registration_day: how long the student had been registered by the cutoff. NaN if unknown or negative. | float64 |
| registration_date_missing | registration | 1 if OULAD has no registration day for this enrolment. | int8 |
| n_assessments_due | assessment | Assessments of this presentation with a known due date on or before the cutoff. 0 for presentations whose first assessment falls later. | int16 |
| n_assessments_submitted | assessment | Of those, how many the student had submitted on or before the cutoff (banked results included). | int16 |
| n_assessments_banked | assessment | Subset of the above carried over from a previous attempt (is_banked = 1). | int16 |
| n_assessments_graded | assessment | Submissions with a non-missing score available by the cutoff. | int16 |
| n_assessments_unsubmitted | assessment | n_assessments_due - n_assessments_submitted, i.e. missed by the cutoff. NOT imputed with a score of zero. | int16 |
| assessment_submission_rate | assessment | n_assessments_submitted / n_assessments_due. NaN (not 0) when no assessment was due. | float64 |
| assessment_graded_rate | assessment | n_assessments_graded / n_assessments_due. NaN when no assessment was due. | float64 |
| n_late_submissions | assessment | Non-banked submissions handed in after their due date but on/before the cutoff. | int16 |
| late_submission_rate | assessment | n_late_submissions / non-banked submissions available by the cutoff. NaN when there are none. | float64 |
| mean_submission_delay_days | assessment | Mean of (date_submitted - due_date) over non-banked submissions; negative = early. | float64 |
| max_submission_delay_days | assessment | Worst observed lateness in days among those submissions. | float64 |
| days_since_last_submission | assessment | cutoff_day - latest submission day observed by the cutoff. NaN when nothing was submitted. | float64 |
| mean_score | assessment | Mean score (0-100) over assessments graded by the cutoff. NaN when nothing was graded. | float64 |
| min_score | assessment | Lowest score graded by the cutoff. | float64 |
| max_score | assessment | Highest score graded by the cutoff. | float64 |
| std_score | assessment | Population std of scores; NaN when fewer than two assessments were graded. | float64 |
| weighted_mean_score | assessment | Score weighted by OULAD 'weight', normalised WITHIN assessment type (see report: OULAD weight is within-type, not a share of the final mark). | float64 |
| first_score | assessment | Score of the earliest-due assessment graded by the cutoff. | float64 |
| latest_score | assessment | Score of the most recently due assessment graded by the cutoff. | float64 |
| score_change | assessment | latest_score - first_score when at least two assessments were graded, else NaN. | float64 |
| has_assessment_due | assessment | 1 if any assessment of this presentation was due by the cutoff. | int8 |
| has_graded_assessment | assessment | 1 if any assessment result was available by the cutoff. | int8 |
| vle_clicks_total | vle | Sum of recorded platform interactions (sum_click) on days <= cutoff, including pre-course days. This is NOT study time in hours. | float64 |
| vle_clicks_precourse | vle | Clicks recorded before day 0 of the presentation. | float64 |
| vle_active_days | vle | Number of distinct days with at least one recorded click. | float64 |
| vle_active_weeks | vle | Number of distinct 7-day blocks (including pre-course weeks) with at least one recorded click. | float64 |
| vle_distinct_sites | vle | Number of distinct VLE resources (id_site) touched. | float64 |
| vle_clicks_per_active_day | vle | vle_clicks_total / vle_active_days. NaN when there is no activity. | float64 |
| vle_activity_group_count | vle | Number of broad resource families touched (forum, content, quiz, collab, wiki, home, other). | int8 |
| vle_clicks_forum | vle | Clicks on forum resources (forumng). | float64 |
| vle_clicks_content | vle | Clicks on content resources (oucontent, resource, subpage, page, url, ...). | float64 |
| vle_clicks_quiz | vle | Clicks on quiz/questionnaire resources. | float64 |
| vle_clicks_collab | vle | Clicks on online collaboration rooms. | float64 |
| vle_clicks_wiki | vle | Clicks on wiki/dataplus resources. | float64 |
| vle_clicks_home | vle | Clicks on the module homepage. | float64 |
| vle_clicks_other | vle | Clicks on resource types outside the six named groups. | float64 |
| vle_forum_share | vle | vle_clicks_forum / vle_clicks_total. NaN when total clicks are 0. | float64 |
| vle_clicks_last7d | vle | Clicks during (cutoff-7, cutoff]. | float64 |
| vle_clicks_prev7d | vle | Clicks during (cutoff-14, cutoff-7]. | float64 |
| vle_click_change_last7 | vle | vle_clicks_last7d - vle_clicks_prev7d; short-term engagement trend. | float64 |
| vle_click_share_last7 | vle | vle_clicks_last7d / vle_clicks_total. NaN when total clicks are 0. | float64 |
| vle_weekly_slope | vle | OLS slope of clicks per course week over weeks 0..k-1 (positive = rising engagement). | float64 |
| vle_days_since_last_activity | vle | cutoff_day - day of last recorded click. Large values indicate disengagement. NaN when never active. | float64 |
| has_vle_activity | vle | 1 if the student has any recorded interaction by the cutoff. Absence of rows in studentVle means zero recorded activity (the log is the complete record of the platform). | int8 |
| vle_clicks_wk0 | vle | Recorded clicks during course week 0 (days 0-6). Week 0 begins on the first day of the presentation. | float64 |
| vle_clicks_wk1 | vle | Recorded clicks during course week 1 (days 7-13). Week 0 begins on the first day of the presentation. | float64 |
| vle_clicks_wk2 | vle | Recorded clicks during course week 2 (days 14-20). Week 0 begins on the first day of the presentation. | float64 |
| vle_clicks_wk3 | vle | Recorded clicks during course week 3 (days 21-27). Week 0 begins on the first day of the presentation. | float64 |
| final_result | target | OULAD-supplied final outcome (Pass/Fail/Withdrawn/Distinction). TARGET SOURCE — never a predictor. | object |
| target_at_risk | target | Derived binary label: 1 if final_result is Fail or Withdrawn, else 0. Derived by this pipeline, not supplied by OULAD. | int8 |
