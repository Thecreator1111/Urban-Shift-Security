# Importing necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib as jb

# Loading the dataset
df = pd.read_csv("data/processed/corridor_safety_data.csv")

X = df.drop(columns=["risk_code", "corridor_id"])
y = df["risk_code"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the models
KNC = KNeighborsClassifier(n_neighbors=5)
KNC.fit(X_train, y_train)

MLP = MLPClassifier(hidden_layer_sizes=(100,100), max_iter=1000, random_state=42)
MLP.fit(X_train, y_train)

DTC = DecisionTreeClassifier(random_state=42)
DTC.fit(X_train, y_train)

# Evaluating the models
knn_acc = KNC.score(X_test, y_test)
mlp_acc = MLP.score(X_test, y_test)
dtc_acc = DTC.score(X_test, y_test)

print("Accuracy of KNC:", knn_acc)
print("Accuracy of MLP:", mlp_acc)
print("Accuracy of DTC:", dtc_acc)

# Determining the best model based on accuracy
if knn_acc >= mlp_acc and knn_acc >= dtc_acc:
    winner_model = KNC
elif mlp_acc >= knn_acc and mlp_acc >= dtc_acc:
    winner_model = MLP
elif dtc_acc >= knn_acc and dtc_acc >= mlp_acc:
    winner_model = DTC
else:
    winner_model = None

# Saving the best model to a file
if winner_model is not None:
    jb.dump(winner_model, "model/safety_model.pkl")

# Creating interactive CLI program to predict the risk code for a given corridor based on its features



def predict_risk_code():
    # Load the saved model
    model = jb.load("model/safety_model.pkl")

    # Validation to ensure the user inputs are valid
    try:
        # Get user input for corridor features
        corr_id = input("Please enter the corridor ID: ")
        hour_of_day = int(input("Please enter the hour of the day[0-23]: "))
        ambient_light = int(input("Please enter the ambient light level[0 - 1]: "))
        street_lighting_condition = int(input("Please enter the street lighting condition[0 - 1]: "))
        crime_risk_index = int(input("Please enter the crime risk index[0 - 10]: "))
        foot_traffic_volume = int(input("Please enter the foot traffic volume[0 - 100]: "))
    except ValueError:
        print("Invalid input. Please enter numeric values for the feature.")
        return

    ambient_light = ambient_light / 100.0
    street_lighting_condition = street_lighting_condition / 100.0
    crime_risk_index = crime_risk_index / 10.0
    foot_traffic_volume = foot_traffic_volume / 100.0

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        "hour_of_day": [hour_of_day],
        "ambient_light": [ambient_light],
        "street_lighting_condition": [street_lighting_condition],
        "crime_risk_index": [crime_risk_index],
        "foot_traffic_volume": [foot_traffic_volume]
    })
    
    # Make prediction
    prediction = model.predict(input_data)
    
    print(f"Predicted Risk Code of the {corr_id} corridor: {prediction[0]}")

if __name__ == "__main__":
    predict_risk_code()