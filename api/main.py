import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import ShiftSafetyRequest, ShiftSafetyResponse

# 1. Initialize FastAPI Application
app = FastAPI(
    title="Urban Shift Safety API",
    description="AI-driven safety classifier and worker shift scheduling backend.",
    version="1.0.0",
)

# 2. Configure CORS (Allows Lovable UI frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Load Trained ML Model Artifact
MODEL_PATH = "model/safety_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print(f"SUCCESS: Loaded model from {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"WARNING: Could not load model from {MODEL_PATH}. Error: {e}")


# 4. Root Health Check Endpoint
# Root route for health check / ping
@app.get("/")
def read_root():
    return {
        "status": "online",
        "project": "Urban Shift Safety API",
        "message": "Backend is active and ready for predictions.",
    }


# 5. Safety Assessment & Risk Prediction Endpoint
@app.post("/predict_safety", response_model=ShiftSafetyResponse)
def predict_safety(payload: ShiftSafetyRequest):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Machine learning model is not loaded on the server.",
        )

    # Convert Pydantic model to DataFrame matching ML feature order
    input_data = pd.DataFrame(
        [
            {
                "hour_of_day": payload.hour_of_day,
                "ambient_light": payload.ambient_light,
                "street_light_func": payload.street_light_func,
                "crime_risk_index": payload.crime_risk_index,
                "foot_traffic": payload.foot_traffic,
            }
        ]
    )

    # Run inference
    risk_code = int(model.predict(input_data)[0])

    # Map Risk Code to Operational Action Plan
    risk_mapping = {
        0: {
            "status": "Safe Corridor",
            "badge_color": "green",
            "action_plan": "Standard single-worker dispatch approved. Proceed with scheduled shift.",
        },
        1: {
            "status": "Group / Escort Needed",
            "badge_color": "orange",
            "action_plan": "Dispatch workers in minimum groups of 4 with high-visibility gear and supervisor check-ins.",
        },
        2: {
            "status": "Unsafe - Shift Suspended",
            "badge_color": "red",
            "action_plan": "Halt solo operations immediately. Re-route shift or assign dedicated mobile escort unit.",
        },
    }

    res = risk_mapping.get(
        risk_code,
        {
            "status": "Unknown Risk",
            "badge_color": "gray",
            "action_plan": "Manual supervisor evaluation required.",
        },
    )

    return ShiftSafetyResponse(
        corridor_id=payload.corridor_id,
        status=res["status"],
        risk_code=risk_code,
        badge_color=res["badge_color"],
        action_plan=res["action_plan"],
    )