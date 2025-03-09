import streamlit as st

st.set_page_config(
    page_title="Multi-Page App",
    page_icon="📊",
    layout="wide",
)

st.sidebar.title("🏠 Home")
st.sidebar.write("Chọn ứng dụng từ sidebar!")

st.title("🎯 Welcome to Multi-Page Streamlit App")
st.write("👉 Sử dụng sidebar để chọn ứng dụng bạn muốn chạy.")

st.markdown("DagsHub: https://dagshub.com/huydfdcv/my-first-repo.mlflow")