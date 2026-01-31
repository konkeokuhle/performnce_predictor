import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance.csv")

df = pd.read_csv(DATA_PATH)

# Separate features and target
X = df.drop("at_risk", axis=1)
y = df["at_risk"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("\n--- MODEL EVALUATION ---")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "student_risk_model.pkl")

print("\n Model trained and saved successfully")
