# Urban Shift Safety & Optimal Scheduler (Addis Ababa)

An AI-driven worker safety classifier and shift scheduling tool for sanitation workers in Addis Ababa corridors.

## Architecture
Lovable UI (React) <---> FastAPI Backend <---> Scikit-Learn Model (.pkl)

## Project Structure
- `data/`: Raw and processed synthetic corridor metrics
- `model/`: Trained `safety_model.pkl` weights
- `api/`: FastAPI server and Pydantic schemas (`api/schemas.py`)
- `src/`: Data generation, validation, and ML training scripts

## Quickstart
1. Install requirements: `pip install -r requirements.txt`
2. Start API backend: `uvicorn api.main:app --reload --port 8000`