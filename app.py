import streamlit as st
import pandas as pd
import joblib

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Prediksi Risiko Diabetes", page_icon="🩺", layout="centered")

# 1. PERSIAPAN MUAT MODEL DENGAN CACHE
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.joblib')

model = load_model()

# 2. UI STREAMLIT
st.title("🩺 Prediksi Risiko Diabetes Pasien")
st.write("Aplikasi dasbor interaktif untuk memprediksi risiko diabetes berdasarkan parameter kesehatan.")

st.markdown("---")

# Layout form input 2 kolom
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
    age = st.number_input("Usia (Age)", min_value=0.0, max_value=120.0, value=35.0)
    hypertension = st.selectbox("Hipertensi", [0, 1], format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")
    heart_disease = st.selectbox("Penyakit Jantung", [0, 1], format_func=lambda x: "Tidak (0)" if x == 0 else "Ya (1)")

with col2:
    smoking_history = st.selectbox("Riwayat Merokok", ["never", "No Info", "current", "former", "not current", "pervious"])
    bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=10.0, max_value=100.0, value=25.0)
    hba1c_level = st.number_input("Kadar HbA1c", min_value=3.0, max_value=15.0, value=5.5)
    blood_glucose_level = st.number_input("Kadar Glukosa Darah", min_value=50.0, max_value=300.0, value=100.0)

st.markdown("---")

# 3. LOGIKA PREDIKSI
if st.button("Prediksi Risiko Diabetes", type="primary"):
    # Bungkus input menjadi DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'smoking_history': [smoking_history],
        'bmi': [bmi],
        'HbA1c_level': [hba1c_level],
        'blood_glucose_level': [blood_glucose_level]
    })
    
    # Lakukan prediksi
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0][1]
    
    # Tampilkan hasil
    if prediction == 1:
        st.error(f"⚠️ **Hasil Prediksi: Positif Berisiko Diabetes** (Probabilitas: {prediction_proba * 100:.2f}%)")
        st.warning("Saran: Konsultasikan dengan tenaga medis profesional untuk pemeriksaan lanjutan.")
    else:
        st.success(f"✅ **Hasil Prediksi: Negatif / Risiko Rendah Diabetes** (Probabilitas Risiko: {prediction_proba * 100:.2f}%)")
        st.info("Pertahankan pola hidup sehat dan rutin berolahraga!")