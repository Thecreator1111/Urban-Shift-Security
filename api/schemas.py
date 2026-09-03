from pydantic import BaseModel, Field

class ShiftSafetyRequest(BaseModel):
    corridor_id: str = Field(..., example="Megenagna")
    hour_of_day: int = Field(..., ge=0, le=23, example=22)
    ambient_light: int = Field(..., ge=0.0, le=1.0, example=0.1)
    street_light_func: int = Field(..., ge=0.0, le=1.0, example=0.4)
    crime_risk_index: int = Field(..., ge=0.0, le=10.0, example=6.5)
    foot_traffic: int = Field(..., ge=0.0, le=100.0, example=15.0)

class ShiftSafetyResponse(BaseModel):
    corridor_id: str = Field(..., json_schema_extra={"example": "Megenagna"})
    status: str = Field(..., example="Group / Escort Needed")
    risk_code: int = Field(..., example=1)
    badge_color: str = Field(..., example="orange")
    action_plan: str = Field(..., example="Dispatch workers in minimum groups of 4 with high-vis equipment.")