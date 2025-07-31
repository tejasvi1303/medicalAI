import joblib

mlb = joblib.load("mlb.pkl")
print("Total known symptoms:", len(mlb.classes_))
print("Sample symptoms:", mlb.classes_[:20])  # print first 20
