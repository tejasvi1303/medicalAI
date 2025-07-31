import os
import joblib

# Paths to your saved objects
MLB_PATH = "mlb.pkl"
MODEL_PATH = "disease_model.pkl"  # your actual model file

# Globals to cache loaded objects
mlb = None
model = None

def _load_resources():
    global mlb, model
    if mlb is None:
        if not os.path.exists(MLB_PATH):
            raise FileNotFoundError(f"MultiLabelBinarizer file not found at {MLB_PATH}")
        mlb = joblib.load(MLB_PATH)
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

def _generate_advice_for(disease_label):
    # Placeholder mapping; extend this with real medical advice logic
    advice_map = {
        "flu": "Rest, hydrate, and take acetaminophen if needed.",
        "cold": "Stay warm, drink fluids, and rest.",
        "migraine": "Avoid bright lights, rest in a quiet room, consider over-the-counter pain relief.",
        # add more disease-to-advice mappings based on your label set
    }
    return advice_map.get(disease_label.lower(), "Consult a physician for detailed advice.")

def predict_disease(user_input_text):
    """
    Returns (disease, advice) on success, or ("", error_message) on failure.
    """
    try:
        _load_resources()
    except Exception as e:
        return "", f"Resource loading error: {e}"

    # Normalize and split symptoms
    user_symptoms = [s.strip().lower() for s in user_input_text.split(",") if s.strip()]

    try:
        known_symptoms = set(mlb.classes_)
    except Exception as e:
        return "", f"Error accessing mlb classes_: {e}"

    valid_symptoms = [symptom for symptom in user_symptoms if symptom in known_symptoms]

    if not valid_symptoms:
        return "", "None of the provided symptoms matched the trained symptom list."

    try:
        user_input_encoded = mlb.transform([valid_symptoms])
        prediction = model.predict(user_input_encoded)[0]
        advice = _generate_advice_for(prediction)
        return prediction, advice
    except Exception as e:
        return "", f"Prediction error: {e}"
