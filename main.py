from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Inisialisasi FastAPI
app = FastAPI(title="Breast Cancer Prediction API")

# Load model dan scaler
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Skema input untuk prediksi kanker (tanpa diagnosis)
class Patient(BaseModel):
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    radius_se: float
    perimeter_se: float
    area_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    texture_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float



# Fungsi untuk preprocessing input data
def preprocess_input(data: Patient):
    # Buat dataframe awal dari input user
    df = pd.DataFrame([{
            "texture_mean": data.texture_mean,
            "perimeter_mean": data.perimeter_mean,
            "area_mean": data.area_mean,
            "smoothness_mean": data.smoothness_mean,
            "compactness_mean": data.compactness_mean,
            "concavity_mean": data.concavity_mean,
            "concave points_mean": data.concave_points_mean,
            "symmetry_mean": data.symmetry_mean,
            "radius_se": data.radius_se,
            "perimeter_se": data.perimeter_se,
            "area_se": data.area_se,
            "compactness_se": data.compactness_se,
            "concavity_se": data.concavity_se,
            "concave points_se": data.concave_points_se,
            "texture_worst": data.texture_worst,
            "area_worst": data.area_worst,
            "smoothness_worst": data.smoothness_worst,
            "compactness_worst": data.compactness_worst,
            "concave points_worst": data.concave_points_worst,
            "symmetry_worst": data.symmetry_worst,
            "fractal_dimension_worst": data.fractal_dimension_worst,
        }])

    # Feature engineering: hitung shape_complexity
    df['shape_complexity'] = (
        data.concavity_mean + data.compactness_mean + data.concave_points_mean
    )


    # Normalisasi
    df_scaled = scaler.transform(df)
    return df_scaled


@app.get("/")
def read_root():
    return {"message": "Breast Cancer Prediction API is running"}

@app.post("/predict")
def predict_diagnosis(data: Patient):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    return {
        "prediction": int(prediction),
        "result": result
    }
