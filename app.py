import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# --- 1. Load Model dan Scaler (hasil training) ---
@st.cache_resource
def load_artifacts():
    model = load_model('stroke_prediction_model.h5')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

model, scaler = load_artifacts()

# --- 2. Antarmuka Aplikasi Streamlit ---
st.title('Aplikasi Prediksi Risiko Stroke')
st.write('Masukkan data pasien untuk memprediksi kemungkinan terkena stroke.')

# --- Input Pengguna ---
with st.sidebar:
    st.header('Input Data Pasien')

    gender = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    age = st.slider('Usia (tahun)', 0, 100, 40)
    hypertension = st.selectbox('Hipertensi (Tekanan Darah Tinggi)', ['No', 'Yes'])
    heart_disease = st.selectbox('Penyakit Jantung', ['No', 'Yes'])
    ever_married = st.selectbox('Pernah Menikah', ['No', 'Yes'])
    work_type = st.selectbox(
        'Jenis Pekerjaan',
        ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
    )
    residence_type = st.selectbox('Tipe Tempat Tinggal', ['Urban', 'Rural'])
    avg_glucose_level = st.slider('Rata-rata Kadar Glukosa', 50.0, 300.0, 100.0)
    bmi = st.slider('Indeks Massa Tubuh (BMI)', 10.0, 60.0, 25.0)
    smoking_status = st.selectbox(
        'Status Merokok',
        ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
    )

# --- Preprocessing Input ---
def preprocess_input(gender, age, hypertension, heart_disease, ever_married,
                       work_type, residence_type, avg_glucose_level, bmi,
                       smoking_status, scaler):

    input_data = {
        'gender': gender,
        'age': age,
        'hypertension': 1 if hypertension == 'Yes' else 0,
        'heart_disease': 1 if heart_disease == 'Yes' else 0,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi,
        'smoking_status': smoking_status
    }

    df_input = pd.DataFrame([input_data])

    # Encoding kategori (konsisten dengan training)
    categorical_cols = [
        'gender', 'ever_married', 'work_type',
        'Residence_type', 'smoking_status'
    ]

    gender_options = ['Male', 'Female']
    ever_married_options = ['Yes', 'No']
    work_type_options = [
        'Private', 'Self-employed', 'Govt_job',
        'children', 'Never_worked'
    ]
    residence_type_options = ['Urban', 'Rural']
    smoking_status_options = [
        'formerly smoked', 'never smoked',
        'smokes', 'Unknown'
    ]

    for col, options in zip(
        categorical_cols,
        [gender_options, ever_married_options,
         work_type_options, residence_type_options,
         smoking_status_options]
    ):
        df_input[col] = pd.Categorical(df_input[col], categories=options)

    df_processed = pd.get_dummies(
        df_input,
        columns=categorical_cols,
        drop_first=True
    )

    expected_columns = [
        'age', 'hy
