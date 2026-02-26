import streamlit as st
import pandas as pd
import joblib

# 1. Memuat model yang sudah dilatih dari file .pkl
# Pastikan nama model sesuai dengan yang diunduh dari Colab
model = joblib.load('model_iris.pkl')

# 2. Menyiapkan Judul dan Deskripsi Aplikasi
st.title('Aplikasi Prediksi Bunga Iris 🌸')
st.write("""
Aplikasi ini memprediksi jenis bunga Iris (**Setosa, Versicolor, atau Virginica**) 
berdasarkan ukuran kelopak (petal) dan mahkota (sepal).
""")

# 3. Membuat Sidebar untuk Input Pengguna
st.sidebar.header('Masukkan Parameter Bunga')

def input_user():
    # Membuat slider untuk masing-masing fitur
    sepal_length = st.sidebar.slider('Panjang Sepal (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Lebar Sepal (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Panjang Petal (cm)', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Lebar Petal (cm)', 0.1, 2.5, 0.2)
    
    # Menyimpan input ke dalam dataframe
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    fitur = pd.DataFrame(data, index=[0])
    return fitur

# Memanggil fungsi input
df = input_user()

# 4. Menampilkan Parameter yang Dimasukkan
st.subheader('Parameter Input:')
st.write(df)

# 5. Melakukan Prediksi
prediksi = model.predict(df)
nama_target_iris = ['Setosa', 'Versicolor', 'Virginica']

# 6. Menampilkan Hasil Prediksi
st.subheader('Hasil Prediksi:')
st.success(f"Bunga ini diprediksi sebagai: **{nama_target_iris[prediksi[0]]}**")
