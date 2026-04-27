import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# ── Load artifacts once at startup ──────────────────────────────────────────
gmm      = joblib.load("gmm_model.pkl")
scaler   = joblib.load("scaler.pkl")
metadata = joblib.load("metadata.pkl")

FEATURES  = metadata["features"]   # list of feature names in the correct order
LABEL_MAP = metadata["label_map"]  # {component_index: "Segment Name"}

app = FastAPI(title="Churn Risk Profiling API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ───────────────────────────────────────────────
class UserMetrics(BaseModel):
    avg_weekly_usage_hours: float = Field(..., ge=0, example=8.5)
    support_tickets:        int   = Field(..., ge=0, example=4)
    payment_failures:       int   = Field(..., ge=0, example=1)
    tenure_months:          float = Field(..., ge=0, example=18)
    last_login_days_ago:    float = Field(..., ge=0, example=12)


class SegmentResponse(BaseModel):
    primary_segment:       str
    confidence_pct:        float
    probability_distribution: dict[str, float]   # {"Segment Name": pct, ...}
    model:                 str


# ── Endpoint ─────────────────────────────────────────────────────────────────
@app.post("/api/user_profile", response_model=SegmentResponse)
def user_profile(metrics: UserMetrics):
    # 1. Build feature vector in the same order the model was trained on
    try:
        x_raw = np.array([[getattr(metrics, f) for f in FEATURES]])
    except AttributeError as e:
        raise HTTPException(status_code=422, detail=f"Missing feature: {e}")

    # 2. Standardize  (same scaler fitted during training)
    x_scaled = scaler.transform(x_raw)

    # 3. Soft cluster assignment  ← the key call
    proba = gmm.predict_proba(x_scaled)[0]   # shape (n_components,)

    # 4. Map component indices → human-readable segment names + percentages
    distribution = {
        LABEL_MAP[i]: round(float(proba[i]) * 100, 2)
        for i in range(len(proba))
    }

    # 5. Pick the dominant segment
    primary = max(distribution, key=distribution.get)

    return SegmentResponse(
        primary_segment=primary,
        confidence_pct=distribution[primary],
        probability_distribution=distribution,
        model="GaussianMixture(n_components=3, covariance_type='full')",
    )


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "features": FEATURES}