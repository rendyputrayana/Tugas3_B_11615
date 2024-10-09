import streamlit as st
import pandas as pd
import os
import pickle

st.markdown(
    """
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stButton button {
        background-color: #333333;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        color: #ffffff;
    }
    .stTextInput, .stNumberInput input{
        background-color: #333333;
        color: #ffffff;
        border: 2px solid #ffffff;
    }
    .stSidebar{
        background-color: #000000;
        color: #ffffff;
    }
    .sidebar h3, .stSidebar p, .stSidebar label, .stSidebar .header-text{
        color: #ffffff;
    }
    h1{
        font-family: 'Courier New', sans-serif;
        color: #ffffff;
    }
    .uploadedFile{
        font-family: 'Arial', sans-serif;
        color: #ffffff;
    }
    .center{
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .center img{
        width: 150px;
        height: 100px;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class:"center">
        <img src="https://media.tenor.com/oqJo9GcbgjYAAAAi/welcome-image-server.gif" alt="Welcome Image">  
    </div>        
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: color: #4CAF50; center;'>Prediksi IPK - 11615</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: #0073e6;'>Aplikasi ini berguna unntuk memprediksi IPK berdarasrkan nilai Matematika, Bahasa Inggris, dan Bahasa Indonesia</p>
    """,unsafe_allow_html=True
)
st.sidebar.markdown("<h3 class:'header-text'>Upload File dan Input Nilai</h3>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload File dataset_regeresi_IPK.csv", type=['csv'])


if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.write("<h3 style='text-align :center; color: #0073e6;'>Data yang diupload</h3>",unsafe_allow_html=True)
    st.dataframe(input_data)

    model_directory = r'C:\Users\ASUS\Downloads\Supervised Learning (Praktek)\Model Suprevised Learning (Praktek)\Tugas3_X_YYYYY'
    model_path = os.path.join(model_directory, r'best_model.pkl')

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        scaler = loaded_model[0]
        feature_selector = loaded_model[1]
        GBR_model = loaded_model[2]

        st.sidebar.subheader("Masukkan Nilai")
        mtk1 = st.sidebar.number_input("Nilai Matematika Semester 1.1", 60.0, 100.0)
        mtk2 = st.sidebar.number_input("Nilai Matematika Semester 1.2", 60.0, 100.0)
        mtk3 = st.sidebar.number_input("Nilai Matematika Semester 2.1", 68.0, 100.0)
        mtk4 = st.sidebar.number_input("Nilai Matematika Semester 2.2", 66.0, 100.0)
        ing1 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 1.1", 64.0, 98.0) 
        ing2 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 1.2", 62.0, 99.0)
        ing3 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 2.1", 68.0, 99.0)
        ing4 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 2.2", 68.0, 99.0)
        ind1 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 1.1", 67.0, 98.0)
        ind2 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 1.2", 70.0, 99.0)
        ind3 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 2.1", 65.0, 99.0)
        ind4 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 2.2", 70.0, 100.0)

        st.sidebar.subheader("Masukkan KKM Nilai")
        kkm_ind1 = st.sidebar.number_input("KKM Bahasa Indonesia Semester 1.1", 00.0, 100.0)
        kkm_ind2 = st.sidebar.number_input("KKM Bahasa Indonesia Semester 1.2", 00.0, 100.0)
        kkm_ind3 = st.sidebar.number_input("KKM Bahasa Indonesia Semester 2.1", 00.0, 100.0)
        kkm_ind4 = st.sidebar.number_input("KKM Bahasa Indonesia Semester 2.2", 00.0, 100.0)
        kkm_ing1 = st.sidebar.number_input("KKM Bahasa Inggris Semester 1.1", 00.0, 100.0)
        kkm_ing2 = st.sidebar.number_input("KKM Bahasa Inggris Semester 1.2", 00.0, 100.0)
        kkm_ing3 = st.sidebar.number_input("KKM Bahasa Inggris Semester 2.1", 00.0, 100.0)
        kkm_ing4 = st.sidebar.number_input("KKM Bahasa Inggris Semester 2.2", 00.0, 100.0)
        kkm_mtk1 = st.sidebar.number_input("KKM Matematika Semester 1.1", 00.0, 100.0)
        kkm_mtk2 = st.sidebar.number_input("KKM Matematika Semester 1.2", 00.0, 100.0)
        kkm_mtk3 = st.sidebar.number_input("KKM Matematika Semester 2.1", 00.0, 100.0)
        kkm_mtk4 = st.sidebar.number_input("KKM Matematika Semester 2.2", 00.0, 100.0)

        input_data = [mtk1,mtk2,mtk3,mtk4,ing1,ing2,ing3,ing4,ind1,ind2,ind3,ind4,kkm_mtk1,kkm_mtk2,kkm_mtk3,kkm_mtk4,kkm_ing1,kkm_ing2,kkm_ing3,kkm_ing4,kkm_ind1,kkm_ind2,kkm_ind3,kkm_ind4]

        input_data_scaled = scaler.transform([input_data])
        input_data_selected = feature_selector.transform(input_data_scaled)

        if st.sidebar.button("Prediksi"):
            GBR_model_predict = GBR_model.predict(input_data_selected)
            st.markdown(f"<h3 style='text-align: center; color: #4CAF50;'>Prediksi IPK adalah : {GBR_model_predict[0]:.2f}</h3>",unsafe_allow_html=True)
    else:
        st.error("Model tidak ditemukan")




































