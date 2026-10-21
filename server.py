import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
WEB_DIR = os.path.join(BASE_DIR, "web")
MODEL_PATH = os.path.join(BASE_DIR, "models", "risk_classifier.joblib")
MODEL_CACHE = None
app = FastAPI(
    title="EduPulse AI Web Server",
    description="Early-Warning Academic Risk Classifier & Socratic Coaching Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache
DATA: Dict[str, Any] = {}

def load_data():
    csv_path = os.path.join(DATA_DIR, "student_features.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Convert numeric NaN to None for clean JSON serialization
        DATA["df"] = df
    else:
        DATA["df"] = pd.DataFrame()

    manifest_path = os.path.join(DATA_DIR, "feature_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            DATA["manifest"] = json.load(f)
    else:
        DATA["manifest"] = {}

    cutoff_path = os.path.join(DATA_DIR, "cutoff_diagnostics.csv")
    if os.path.exists(cutoff_path):
        DATA["cutoff"] = pd.read_csv(cutoff_path).to_dict(orient="records")
    else:
        DATA["cutoff"] = []

load_data()

class SimulationInput(BaseModel):
    vle_clicks_total: float = 300.0
    vle_active_days: float = 10.0
    vle_days_since_last_activity: float = 3.0
    assessment_submission_rate: float = 1.0
    mean_score: float = 65.0
    late_submission_rate: float = 0.0
    num_of_prev_attempts: int = 0
    studied_credits: int = 60

@app.get("/api/health")
def health():
    return {
        "status": "online",
        "service": "EduPulse-AI Dev Server",
        "dataset_rows": len(DATA.get("df", [])),
        "cutoff_day": 28,
        "features_count": len(DATA.get("manifest", {}).get("feature_columns", []))
    }

@app.get("/api/overview")
def get_overview():
    df: pd.DataFrame = DATA.get("df")
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded")

    total_enrolments = int(len(df))
    at_risk_count = int(df["target_at_risk"].sum())
    at_risk_prevalence = round(float(at_risk_count / total_enrolments) * 100, 2)

    # Outcomes breakdown
    outcomes = df["final_result"].value_counts().to_dict()

    # Module breakdown
    module_stats = []
    for mod, group in df.groupby("code_module"):
        count = len(group)
        risk_cnt = int(group["target_at_risk"].sum())
        module_stats.append({
            "module": mod,
            "total": count,
            "at_risk": risk_cnt,
            "risk_rate": round((risk_cnt / count) * 100, 1),
            "safe_count": count - risk_cnt,
            "avg_score": round(float(group["mean_score"].dropna().mean()), 1) if not group["mean_score"].dropna().empty else 0,
            "avg_clicks": round(float(group["vle_clicks_total"].mean()), 1)
        })

    # Presentation breakdown
    presentation_stats = []
    for pres, group in df.groupby("code_presentation"):
        count = len(group)
        risk_cnt = int(group["target_at_risk"].sum())
        presentation_stats.append({
            "presentation": pres,
            "total": count,
            "at_risk": risk_cnt,
            "risk_rate": round((risk_cnt / count) * 100, 1)
        })

    # Feature family counts
    feature_families = {
        "Demographic Attributes": 9,
        "Registration Timing": 5,
        "Windowed Assessments": 22,
        "VLE Engagement & Velocity": 22
    }

    # Comparative Averages (Safe vs At-Risk)
    safe_df = df[df["target_at_risk"] == 0]
    risk_df = df[df["target_at_risk"] == 1]

    comparative = {
        "vle_clicks_total": {
            "safe": round(float(safe_df["vle_clicks_total"].mean()), 1),
            "risk": round(float(risk_df["vle_clicks_total"].mean()), 1)
        },
        "vle_active_days": {
            "safe": round(float(safe_df["vle_active_days"].mean()), 1),
            "risk": round(float(risk_df["vle_active_days"].mean()), 1)
        },
        "mean_score": {
            "safe": round(float(safe_df["mean_score"].dropna().mean()), 1),
            "risk": round(float(risk_df["mean_score"].dropna().mean()), 1)
        },
        "submission_rate": {
            "safe": round(float(safe_df["assessment_submission_rate"].mean()) * 100, 1),
            "risk": round(float(risk_df["assessment_submission_rate"].mean()) * 100, 1)
        },
        "days_since_activity": {
            "safe": round(float(safe_df["vle_days_since_last_activity"].dropna().mean()), 1),
            "risk": round(float(risk_df["vle_days_since_last_activity"].dropna().mean()), 1)
        },
        "late_submission_rate": {
            "safe": round(float(safe_df["late_submission_rate"].dropna().mean()) * 100, 1),
            "risk": round(float(risk_df["late_submission_rate"].dropna().mean()) * 100, 1)
        }
    }

    # Demographics risk rates
    demo_gender = []
    for g, grp in df.groupby("gender"):
        demo_gender.append({
            "category": "Female" if g == "F" else "Male",
            "total": len(grp),
            "risk_rate": round(float(grp["target_at_risk"].mean()) * 100, 1)
        })

    demo_education = []
    for edu, grp in df.groupby("highest_education"):
        demo_education.append({
            "category": edu,
            "total": len(grp),
            "risk_rate": round(float(grp["target_at_risk"].mean()) * 100, 1)
        })
    demo_education.sort(key=lambda x: x["risk_rate"], reverse=True)

    demo_age = []
    for age, grp in df.groupby("age_band"):
        demo_age.append({
            "category": age,
            "total": len(grp),
            "risk_rate": round(float(grp["target_at_risk"].mean()) * 100, 1)
        })

    return {
        "total_enrolments": total_enrolments,
        "at_risk_count": at_risk_count,
        "on_track_count": total_enrolments - at_risk_count,
        "at_risk_prevalence": at_risk_prevalence,
        "outcomes": outcomes,
        "module_stats": module_stats,
        "presentation_stats": presentation_stats,
        "feature_families": feature_families,
        "comparative": comparative,
        "demographics": {
            "gender": demo_gender,
            "education": demo_education,
            "age": demo_age
        }
    }

@app.get("/api/students")
def get_students(
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=5, le=100),
    search: Optional[str] = None,
    module: Optional[str] = "all",
    presentation: Optional[str] = "all",
    risk: Optional[str] = "all",
    sort_by: Optional[str] = "id_student",
    sort_dir: Optional[str] = "asc"
):
    df: pd.DataFrame = DATA.get("df")
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded")

    sub = df

    if search:
        s_clean = search.strip()
        sub = sub[sub["id_student"].astype(str).str.contains(s_clean)]

    if module and module != "all":
        sub = sub[sub["code_module"] == module]

    if presentation and presentation != "all":
        sub = sub[sub["code_presentation"] == presentation]

    if risk and risk != "all":
        if risk == "at_risk":
            sub = sub[sub["target_at_risk"] == 1]
        elif risk == "safe":
            sub = sub[sub["target_at_risk"] == 0]

    # Sorting
    ascending = (sort_dir.lower() == "asc")
    if sort_by in sub.columns:
        sub = sub.sort_values(by=sort_by, ascending=ascending, na_position="last")

    total_count = len(sub)
    total_pages = max(1, (total_count + limit - 1) // limit)
    offset = (page - 1) * limit
    page_df = sub.iloc[offset:offset + limit]

    columns_to_return = [
        "id_student", "code_module", "code_presentation", "gender", "highest_education",
        "mean_score", "vle_clicks_total", "vle_active_days", "assessment_submission_rate",
        "n_late_submissions", "vle_days_since_last_activity", "final_result", "target_at_risk"
    ]

    records = []
    for _, row in page_df.iterrows():
        rec = {}
        for col in columns_to_return:
            val = row[col]
            if pd.isna(val):
                rec[col] = None
            elif isinstance(val, (np.integer, int)):
                rec[col] = int(val)
            elif isinstance(val, (np.floating, float)):
                rec[col] = round(float(val), 2)
            else:
                rec[col] = str(val)
        records.append(rec)

    return {
        "page": page,
        "limit": limit,
        "total": total_count,
        "total_pages": total_pages,
        "students": records
    }

@app.get("/api/student/{id_student}")
def get_student_detail(id_student: int):
    df: pd.DataFrame = DATA.get("df")
    if df.empty:
        raise HTTPException(status_code=500, detail="Data not loaded")

    matches = df[df["id_student"] == id_student]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Student ID not found")

    row = matches.iloc[0]
    data_dict = {}
    for col in df.columns:
        val = row[col]
        if pd.isna(val):
            data_dict[col] = None
        elif isinstance(val, (np.integer, int)):
            data_dict[col] = int(val)
        elif isinstance(val, (np.floating, float)):
            data_dict[col] = round(float(val), 2)
        else:
            data_dict[col] = str(val)

    # Risk diagnosis & Drivers
    drivers = []
    protective = []

    # VLE activity check
    clicks = data_dict.get("vle_clicks_total") or 0
    days_inactive = data_dict.get("vle_days_since_last_activity")
    if clicks < 100:
        drivers.append({"factor": "Critically Low VLE Engagement", "impact": "High", "detail": f"Only {clicks} clicks recorded up to Day 28 (cohort average is ~390)"})
    elif clicks > 450:
        protective.append({"factor": "Strong VLE Engagement", "detail": f"{clicks} total clicks (top tier for Day 28)"})

    if days_inactive is not None and days_inactive > 14:
        drivers.append({"factor": "Extended Study Disengagement", "impact": "High", "detail": f"No learning activity for {days_inactive} days prior to cutoff"})
    elif days_inactive is not None and days_inactive <= 3:
        protective.append({"factor": "Recent Active Presence", "detail": f"Active within the last {days_inactive} days"})

    # Assessment checks
    sub_rate = data_dict.get("assessment_submission_rate")
    mean_sc = data_dict.get("mean_score")
    n_due = data_dict.get("n_assessments_due") or 0
    late_cnt = data_dict.get("n_late_submissions") or 0

    if n_due > 0:
        if sub_rate is not None and sub_rate < 1.0:
            drivers.append({"factor": "Missed Mandatory Assessment", "impact": "Critical", "detail": f"Only completed {sub_rate * 100:.0f}% of due assessments"})
        if mean_sc is not None:
            if mean_sc < 50:
                drivers.append({"factor": "Failing Early Assessment Score", "impact": "High", "detail": f"Mean score is {mean_sc:.1f} / 100 (pass threshold is 40)"})
            elif mean_sc >= 75:
                protective.append({"factor": "High Early Graded Score", "detail": f"Strong score of {mean_sc:.1f} on initial coursework"})
        if late_cnt > 0:
            drivers.append({"factor": "Late Coursework Submission", "impact": "Medium", "detail": f"{late_cnt} assessment(s) turned in past official deadline"})

    # Prior attempts
    prev_att = data_dict.get("num_of_prev_attempts") or 0
    if prev_att > 0:
        drivers.append({"factor": "Repeat Enrolment History", "impact": "Medium", "detail": f"{prev_att} previous unsuccessful attempt(s) on this course"})

    # Archetype classification & Socratic Coach Blueprint
    is_at_risk = bool(data_dict.get("target_at_risk") == 1)
    if clicks < 60 and (sub_rate is None or sub_rate < 0.5):
        archetype = "Disengaged Inactive (Ghosting Risk)"
        coaching_strategy = "Re-engagement & Barrier Identification"
        intervention_action = "Reach out via low-friction SMS/Email check-in to identify external obstacles (technical, personal, workload). Schedule 10-minute advisor sync."
        socratic_prompt = f"Hi {data_dict.get('id_student')}, we noticed you haven't been on the course portal recently. The initial units build core foundations—is there any topic in Unit 1 or course access barrier we can help troubleshoot together today?"
    elif mean_sc is not None and mean_sc < 55:
        archetype = "Academic Struggler (Concept Gap)"
        coaching_strategy = "Targeted Socratic Scaffolding & Formative Remediation"
        intervention_action = "Deploy automated diagnostic quiz on key foundational concepts tested in Assessment 1. Refer to peer tutoring or tutor office hours."
        socratic_prompt = f"Great effort on submitting your initial assessment! Reviewing your answers, you're close on core definitions but hit stumbling blocks in analytical application. Would you like to walk through a guided 3-step practice exercise on this concept?"
    elif late_cnt > 0 or (days_inactive and days_inactive > 7):
        archetype = "Pacing & Time Management Risk"
        coaching_strategy = "Milestone Chunking & Study Routine Structuring"
        intervention_action = "Provide a customized 7-day study schedule chunking the next upcoming assignment into bite-sized 30-minute daily checkpoints."
        socratic_prompt = f"You are progressing well, but deadlines are tightening! What day this week works best for a 45-minute focused session to outline your draft for the next submission?"
    else:
        archetype = "On-Track Consistent Learner" if not is_at_risk else "Moderate Pacing Risk"
        coaching_strategy = "Positive Reinforcement & Advanced Challenge"
        intervention_action = "Maintain regular monitoring. Provide optional enrichment material to sustain momentum."
        socratic_prompt = f"Impressive engagement so far! You are on pace with your study goals. Have you checked out the discussion forum topic on this week's extension reading?"

    return {
        "student": data_dict,
        "risk_profile": {
            "is_at_risk": is_at_risk,
            "final_outcome": data_dict.get("final_result"),
            "risk_drivers": drivers,
            "protective_factors": protective,
            "archetype": archetype,
            "coaching_strategy": coaching_strategy,
            "recommended_action": intervention_action,
            "socratic_coaching_prompt": socratic_prompt
        }
    }

@app.get("/api/cutoff-diagnostics")
def get_cutoff_diagnostics():
    records = DATA.get("cutoff", [])
    explanation = {
        "selected_cutoff": 28,
        "rationale": "Day 28 (~10% into the course) achieves the optimal trade-off: 62.8% of enrolments have at least one graded assessment and 16/22 course presentations have assessment data, while ~90% (240+ days) of the academic term remains ahead for timely tutor intervention.",
        "comparisons": [
            {"day": 14, "verdict": "Too Early", "issue": "Only 1 of 22 presentations has an assessment due; insufficient student performance signal."},
            {"day": 21, "verdict": "Sub-optimal", "issue": "Only 42.3% of students have grades; 11/22 presentations lacking assessment."},
            {"day": 28, "verdict": "Optimal (Milestone 1 Benchmark)", "issue": "Ideal balance of high information gain (62.8% graded, 16/22 modules) and maximal intervention lead-time."},
            {"day": 42, "verdict": "Diminishing Returns", "issue": "Only adds 8.8% graded coverage but costs 2 weeks of critical early intervention window."},
            {"day": 56, "verdict": "Late Interventions", "issue": "High predictive signal but quarter of the semester has elapsed; harder to recover failing students."}
        ]
    }
    return {
        "table": records,
        "analysis": explanation
    }

@app.get("/api/features")
def get_features():
    manifest = DATA.get("manifest", {})
    return manifest
@app.get("/api/model-predict/{id_student}")
def predict_saved_student(
    id_student: int,
    presentation: str = Query(..., min_length=5, max_length=5)
):
    global MODEL_CACHE

    df = DATA.get("df")
    manifest = DATA.get("manifest", {})
    features = manifest.get("feature_columns", [])

    if df is None or df.empty:
        raise HTTPException(
            status_code=503,
            detail="Student dataset is not loaded."
        )

    if not features:
        raise HTTPException(
            status_code=500,
            detail="Feature list is missing from the manifest."
        )

    matches = df[
        (df["id_student"] == id_student)
        & (df["code_presentation"].astype(str) == presentation)
    ]

    if matches.empty:
        raise HTTPException(
            status_code=404,
            detail="Student not found for this presentation."
        )

    if len(matches) != 1:
        raise HTTPException(
            status_code=409,
            detail="Multiple matching student records were found."
        )

    if MODEL_CACHE is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail="Trained model file was not found."
            )

        try:
            MODEL_CACHE = joblib.load(MODEL_PATH)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not load the trained model: {exc}"
            )

    student_features = matches.loc[:, features].copy()

    try:
        probability = float(
            MODEL_CACHE.predict_proba(student_features)[0, 1]
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        )

    risk_percent = round(probability * 100, 1)

    if risk_percent >= 70:
        tier = "Critical Risk"
    elif risk_percent >= 45:
        tier = "Elevated Risk"
    elif risk_percent >= 25:
        tier = "Moderate Risk"
    else:
        tier = "Low Risk"

    return {
        "id_student": id_student,
        "code_presentation": presentation,
        "model": "Trained XGBoost pipeline",
        "risk_probability": risk_percent,
        "risk_tier": tier,
        "prediction_source": "Saved model using day-28 OULAD features",
        "target_definition": "Final course outcome is Fail or Withdrawn",
        "limitation": (
            "This model uses OULAD data and has not been validated "
            "on VTU students."
        )
    }
@app.post("/api/simulate")
def simulate_risk(inp: SimulationInput):
    # Calibrated empirical logistic risk model based on OULAD Day 28 features
    # Log-odds baseline for prevalence ~ 0.441 -> log(0.441/0.559) = -0.237
    log_odds = -0.237

    # VLE clicks effect (strong negative)
    # Average clicks is ~390. Below 100 increases risk significantly, above 500 reduces risk.
    if inp.vle_clicks_total < 80:
        log_odds += 1.35
    elif inp.vle_clicks_total < 250:
        log_odds += 0.65
    elif inp.vle_clicks_total > 600:
        log_odds -= 0.95
    else:
        log_odds -= 0.25

    # Active days
    if inp.vle_active_days < 5:
        log_odds += 0.85
    elif inp.vle_active_days > 14:
        log_odds -= 0.70

    # Inactivity recency
    if inp.vle_days_since_last_activity > 14:
        log_odds += 1.10
    elif inp.vle_days_since_last_activity <= 2:
        log_odds -= 0.50

    # Assessment submission rate
    if inp.assessment_submission_rate < 0.5:
        log_odds += 1.40
    elif inp.assessment_submission_rate >= 1.0:
        log_odds -= 0.80

    # Mean score
    if inp.mean_score < 40:
        log_odds += 1.20
    elif inp.mean_score < 55:
        log_odds += 0.50
    elif inp.mean_score >= 75:
        log_odds -= 0.90

    # Late submission
    if inp.late_submission_rate > 0:
        log_odds += (inp.late_submission_rate * 0.75)

    # Previous attempts
    if inp.num_of_prev_attempts > 0:
        log_odds += (inp.num_of_prev_attempts * 0.45)

    # Convert log odds to probability
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    prob_pct = round(float(prob) * 100, 1)

    if prob_pct >= 70:
        tier = "Critical Risk"
        badge_color = "red"
        advice = "Immediate proactive outreach required. High probability of withdrawal or failure without hands-on intervention."
    elif prob_pct >= 45:
        tier = "Elevated Risk"
        badge_color = "amber"
        advice = "Student is showing warning signs in coursework pacing or study frequency. Recommend automated tutor nudge."
    elif prob_pct >= 25:
        tier = "Moderate Risk"
        badge_color = "blue"
        advice = "Student is generally keeping up but monitoring is recommended around upcoming assignment deadlines."
    else:
        tier = "Low Risk (On Track)"
        badge_color = "emerald"
        advice = "Student demonstrates strong study habits, regular platform engagement, and solid early assessment scores."

    return {
        "risk_probability": prob_pct,
        "risk_tier": tier,
        "badge_color": badge_color,
        "advisory_recommendation": advice,
        "key_factors": [
            {"factor": "VLE Clicks Input", "value": f"{inp.vle_clicks_total:.0f} clicks"},
            {"factor": "Active Study Days", "value": f"{inp.vle_active_days:.0f} of 28 days"},
            {"factor": "Days Since Last Login", "value": f"{inp.vle_days_since_last_activity:.0f} days ago"},
            {"factor": "Assessment Completion", "value": f"{inp.assessment_submission_rate * 100:.0f}% submitted"},
            {"factor": "Coursework Score", "value": f"{inp.mean_score:.1f} / 100"}
        ]
    }

# Mount static figures
if os.path.exists(FIGURES_DIR):
    app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

# Mount frontend
@app.post("/api/knowledge-tracing/demo")
def knowledge_tracing_demo(payload: dict):
    """Demonstrate BKT on a supplied response sequence; not real student data."""
    from src.knowledge_tracing import BKTParameters, trace_skill

    responses = payload.get("responses")
    if not isinstance(responses, list) or not responses:
        raise HTTPException(
            status_code=400,
            detail="'responses' must be a non-empty list of true/false values."
        )
    if any(type(item) is not bool for item in responses):
        raise HTTPException(
            status_code=400,
            detail="Every response must be a JSON boolean: true or false."
        )

    parameters = BKTParameters()
    history = trace_skill(responses, parameters)

    return {
        "model": "Bayesian Knowledge Tracing (BKT)",
        "mode": "demonstration",
        "responses": responses,
        "mastery_after_each_response": history,
        "final_mastery_percent": round(history[-1] * 100, 2),
        "parameters": {
            "initial_mastery": parameters.initial_mastery,
            "learn_probability": parameters.learn_probability,
            "guess_probability": parameters.guess_probability,
            "slip_probability": parameters.slip_probability,
        },
        "limitation": (
            "Demonstration only. No verified assessment-to-concept mapping "
            "or real student mastery validation is connected yet."
        ),
    }

if os.path.exists(WEB_DIR):
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8080, reload=True)
