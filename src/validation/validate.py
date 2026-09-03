import pandas as pd

def validate():
    df = pd.read_csv("data/processed/corridor_safety_data.csv")
    
    # Assert quality gates
    assert not df.isnull().values.any(), "Data contains null values!"
    assert len(df) >= 500, "Dataset too small!"
    assert set(df["risk_code"].unique()).issubset({0, 1, 2}), "Invalid risk codes!"
    assert df["hour_of_day"].between(0, 23).all(), "Invalid hours!"
    assert df["ambient_light"].between(0.0, 1.0).all(), "Invalid ambient light!"
    assert df["street_light_func"].between(0.0, 1.0).all(), "Invalid street light!"
    assert df["crime_risk_index"].between(0.0, 10.0).all(), "Invalid crime index!"
    assert df["foot_traffic"].between(0.0, 100.0).all(), "Invalid foot traffic!"

    print("PASSED: All Data Quality Gates Verified Successfully.")

if name == "main":
    validate()