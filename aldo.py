import streamlit as st
import pickle
import numpy as np

# Fungsi untuk memuat model, encoder, dan scaler
# def load_model_and_preprocessors(model_name):
#     if model_name == "Stochastic Gradient Descent (SGD)":
#         model = pickle.load(open("sgd_model.pkl", "rb"))
#         encoder = pickle.load(open("enc.pkl", "rb"))
#         scaler = pickle.load(open("sc.pkl", "rb"))
#     elif model_name == "Perceptron":
#         model = pickle.load(open("ppn_model.pkl", "rb"))
#         encoder = pickle.load(open("enc_ppn.pkl", "rb"))
#         scaler = pickle.load(open("sc_ppn.pkl", "rb"))

#     elif model_name == "Support Vector Machine (SVM)":
#         model = pickle.load(open("svm.pkl", "rb"))
#         encoder = pickle.load(open("svm_enc.pkl", "rb"))
#         scaler = pickle.load(open("svm_sc.pkl", "rb"))
#     else:
#         raise ValueError("Model tidak ditemukan!")
#     return model, encoder, scaler
def load_model_and_preprocessors(model_name):
    try:
        if model_name == "Stochastic Gradient Descent (SGD)":
            model = pickle.load(open("sgd_model.pkl", "rb"))
            encoder = pickle.load(open("enc.pkl", "rb"))
            scaler = pickle.load(open("sc.pkl", "rb"))
        elif model_name == "Perceptron":
            model = pickle.load(open("ppn_model.pkl", "rb"))
            encoder = pickle.load(open("enc_ppn.pkl", "rb"))
            scaler = pickle.load(open("sc_ppn.pkl", "rb"))
        elif model_name == "Support Vector Machine (SVM)":
            model = pickle.load(open("svm.pkl", "rb"))
            encoder = pickle.load(open("svm_enc.pkl", "rb"))
            scaler = pickle.load(open("svm_sc.pkl", "rb"))
        else:
            raise ValueError("Model tidak ditemukan!")
        return model, encoder, scaler
    except Exception as e:
        st.error(f"Error loading model/preprocessors: {e}")
        raise e

# Judul aplikasi
st.title("Prediksi Jenis Ikanmu Disini :)")
st.write("Pilih algoritma dan masukkan data untuk prediksi.")

# Dropdown untuk memilih algoritma
algorithm = st.selectbox("Pilih Algoritma:", ["Stochastic Gradient Descent (SGD)", "Perceptron","Support Vector Machine (SVM)"])

# Input data
st.write("### Masukkan Data")
feature1 = st.number_input("Length", min_value=0.0, format="%.2f")
feature2 = st.number_input("Weight", min_value=0.0, format="%.2f")
feature3 = st.number_input("weight-length-ratio", min_value=0.0, format="%.2f")

# Tombol untuk prediksi
if st.button("Prediksi"):
    try:
        # Load model, encoder, dan scaler berdasarkan algoritma yang dipilih
        model, encoder, scaler = load_model_and_preprocessors(algorithm)
        
        # Scaling data input
        input_data = np.array([[feature1, feature2, feature3]])
        input_data_scaled = scaler.transform(input_data)
        
        # Prediksi
        prediction = model.predict(input_data_scaled)
        prediction_label = encoder.inverse_transform(prediction)
        
        # Tampilkan hasil
        st.success(f"Hasil Prediksi: {prediction_label[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
