import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('model_dropout.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Sistem Prediksi Kelulusan Siswa - Jaya Jaya Institut")
st.write("Masukkan data akademik siswa untuk memprediksi status (Dropout/Graduate/Enrolled).")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        course = st.number_input("Course ID", value=33)
        tuition = st.selectbox("Tuition fees up to date", [0, 1])
        scholarship = st.selectbox("Scholarship holder", [0, 1])
        age = st.number_input("Age at enrollment", value=20)

    with col2:
        sem1_grade = st.number_input("Curricular units 1st sem (grade)", value=0.0)
        sem2_grade = st.number_input("Curricular units 2nd sem (grade)", value=0.0)
        sem1_approved = st.number_input("Curricular units 1st sem (approved)", value=0)
        sem2_approved = st.number_input("Curricular units 2nd sem (approved)", value=0)
    
    submit = st.form_submit_button("Prediksi")

if submit:

    input_data = np.zeros((1, 36)) 
    input_data[0, 16] = tuition
    input_data[0, 18] = scholarship
    input_data[0, 19] = age
    input_data[0, 24] = sem1_approved
    input_data[0, 25] = sem1_grade
    input_data[0, 30] = sem2_approved
    input_data[0, 31] = sem2_grade

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    result = le.inverse_transform(prediction)[0]

    st.subheader(f"Hasil Prediksi: {result}")
    if result == "Dropout":
        st.warning("Siswa ini berisiko tinggi dropout. Disarankan bimbingan khusus.")
    else:
        st.success("Siswa ini diprediksi aman.")
