import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train():
    # 1. Load validated dataset
    data_path = "data/processed/corridor_safety_data.csv"
    df = pd.read_csv(data_path)

    # 2. Select numerical feature columns & target label
    feature_cols = [
        "hour_of_day",
        "ambient_light",
        "street_light_func",
        "crime_risk_index",
        "foot_traffic",
    ]
    X = df[feature_cols]
    y = df["risk_code"]

    # 3. Stratified Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)

    # 5. Evaluate Performance
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n--- MODEL EVALUATION REPORT ---")
    print(f"Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=["Safe (0)", "Escort (1)", "Unsafe (2)"]))

    # 6. Export trained model binary artifact
    model_export_path = "model/safety_model.pkl"
    joblib.dump(clf, model_export_path)
    print(f"SUCCESS: Model saved to -> {model_export_path}")

if __name__ == "__main__":
    train()