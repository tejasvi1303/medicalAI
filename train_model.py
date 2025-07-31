import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("DiseaseAndSymptoms.csv").dropna(how="all")

# Identify symptom columns (skip 'Disease')
symptom_columns = [col for col in df.columns if col != "Disease"]

# Clean symptoms
def clean_symptom(symptom):
    if pd.isna(symptom):
        return None
    return str(symptom).strip().lower().replace(" ", "_")

# Combine cleaned symptom lists
df["Symptoms"] = df[symptom_columns].apply(
    lambda row: [clean_symptom(s) for s in row if pd.notna(s)],
    axis=1
)

# Fit encoder
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["Symptoms"])
y = df["Disease"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save files
joblib.dump(model, "disease_model.pkl")
joblib.dump(mlb, "mlb.pkl")

print(f"✅ Model trained with {len(mlb.classes_)} unique symptoms.")
print("Sample symptoms:", mlb.classes_[:20])
