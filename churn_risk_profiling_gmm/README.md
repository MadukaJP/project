# Churn Risk Profiling (GMM)

[![Frontend Live](https://img.shields.io/badge/Frontend-Live%20on%20Netlify-00C7B7?logo=netlify&logoColor=white)](https://churn-risk-profiling.netlify.app)
[![Backend Live](https://img.shields.io/badge/Backend-Live%20API-009688?logo=fastapi&logoColor=white)](https://churn-risk-profiling.fastapicloud.dev)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/jayjoshi37/customer-subscription-churn-and-usage-patterns/data)
[![Model](https://img.shields.io/badge/Model-Gaussian%20Mixture%20Model-orange)](#model-overview)
[![API](https://img.shields.io/badge/API-FastAPI-05998B?logo=fastapi&logoColor=white)](#backend-fastapi)
[![Frontend](https://img.shields.io/badge/Frontend-HTML%20%7C%20Tailwind%20%7C%20Chart.js-3B82F6)](#frontend-dashboard)

Customer churn risk profiling app that segments users into:
- **Power User**
- **Standard User**
- **At-Risk / Churning User**

The system uses a trained **Gaussian Mixture Model (GMM)** backend and a static frontend dashboard for interactive prediction and visualization.

## Tech Stack

- **Backend API:** FastAPI (Python)
- **Frontend:** HTML
- **Styling:** Tailwind CSS (CDN)
- **Visualization:** Chart.js
- **ML/Data:** NumPy, scikit-learn, joblib
- **Serving (local):** Uvicorn

## Live Links

- **Frontend (Netlify):** [https://churn-risk-profiling.netlify.app](https://churn-risk-profiling.netlify.app)
- **Backend API:** [https://churn-risk-profiling.fastapicloud.dev](https://churn-risk-profiling.fastapicloud.dev)
- **API Docs:** [https://churn-risk-profiling.fastapicloud.dev/docs](https://churn-risk-profiling.fastapicloud.dev/dpcs)
- **Dataset (Kaggle):** [Customer Subscription Churn and Usage Patterns](https://www.kaggle.com/datasets/jayjoshi37/customer-subscription-churn-and-usage-patterns/data)

## Project Structure

```text
churn_risk_profiling_gmm/
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── gmm_model.pkl
│   ├── scaler.pkl
│   └── metadata.pkl
├── frontend/
│   └── index.html
├── notebooks/
│   └── Churn_Risk_Profiling_GMM.ipynb
└── readme
```

## Model Overview

- **Algorithm:** Gaussian Mixture Model (GMM)
- **Output:** soft probability distribution across churn-risk segments
- **Primary Segment:** highest-probability segment
- **Confidence:** probability of the primary segment (in %)

Input features used by the model:
- `avg_weekly_usage_hours`
- `support_tickets`
- `payment_failures`
- `tenure_months`
- `last_login_days_ago`

## Backend (FastAPI)

The backend exposes prediction and health endpoints.

### Endpoints

- `POST /api/user_profile`  
  Accepts user metrics and returns segment probabilities plus confidence.
- `GET /health`  
  Returns API health status and model feature list.

### Example Request

```json
{
  "avg_weekly_usage_hours": 9.0,
  "support_tickets": 4,
  "payment_failures": 1,
  "tenure_months": 18,
  "last_login_days_ago": 14
}
```

## Frontend Dashboard

The frontend is a static single-page dashboard (`frontend/index.html`) with:
- form-based input for user behavior metrics
- quick example profile buttons
- probability chart visualization
- primary segment and confidence display
- toast notifications for request status

The frontend currently calls:
- `https://churn-risk-profiling.fastapicloud.dev/api/user_profile`

## Run Locally

### 1) Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Backend local URL: `http://localhost:8000`  
Health check: `http://localhost:8000/health`

### 2) Frontend

Open `frontend/index.html` in your browser, or serve it with a static server.

## Notes

- CORS is currently open in the backend (`allow_origins=["*"]`) for easier integration.
- Ensure `gmm_model.pkl`, `scaler.pkl`, and `metadata.pkl` exist in `backend/` when running locally.
