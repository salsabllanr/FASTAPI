# 🦠🧬 Breast Cancer Prediction API

Sebuah mini-proyek berbasis **FastAPI** yang dapat mengidentifikasi kemungkinan **karakteristik utama tumor jinak dan ganas**.

## 📁 Struktur File

```
├── main.py             # Endpoint API utama
├── model.pkl           # File model Machine Learning yang telah dilatih
├── scaler.pkl          # File scaler untuk normalisasi fitur input
├── requirements.txt    # Daftar dependency yang dibutuhkan
```

## 🚀 Fitur API

- Prediksi jenis tumor
- Menerima input melalui metode POST
- Hasil prediksi: 0 untuk Benign(Jinak), 1 untuk Malignant(Ganas)

## ⚙️ Cara Menjalankan

### 1. Clone Repositori

```bash
git clone https://github.com/salsabllanr/FASTAPI/tree/main/cancer.git
cd cancer
```

### 2. Buat Virtual Environment

```bash
python -m venv .env
source .env/bin/activate  # Command Prompt: .env\Scripts\activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan API

```bash
fastapi dev
```

### 5. Akses Swagger UI

Buka browser ke:  
👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 🧪 Contoh JSON Input

```json
{
  "texture_mean": 19.3,
  "perimeter_mean": 92.0,
  "area_mean": 654.9,
  "smoothness_mean": 0.096,
  "compactness_mean": 0.104,
  "concavity_mean": 0.088,
  "concave_points_mean": 0.048,
  "symmetry_mean": 0.181,
  "radius_se": 0.405,
  "perimeter_se": 2.86,
  "area_se": 40.3,
  "compactness_se": 0.030,
  "concavity_se": 0.037,
  "concave_points_se": 0.012,
  "texture_worst": 25.7,
  "area_worst": 880.6,
  "smoothness_worst": 0.132,
  "compactness_worst": 0.254,
  "concave_points_worst": 0.114,
  "symmetry_worst": 0.290,
  "fractal_dimension_worst": 0.083
}

```

## ✅ Contoh Output

```json
{
  "prediction": 0,
  "result": "Benign"
}
```


> Dibuat sebagai bagian dari praktik tahap **Deployment** dalam metode **CRISP-DM**.  
> Proyek ini dapat dijadikan dasar pengembangan API prediksi sederhana lainnya.
