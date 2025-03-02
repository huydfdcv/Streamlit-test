import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mlflow.tracking import MlflowClient
import os

# Cấu hình MLflow với DagsHub
mlflow_tracking_uri = st.secrets["MLFLOW_TRACKING_URI"]  # Đọc từ Streamlit Secrets
mlflow_token = st.secrets["MLFLOW_TRACKING_TOKEN"]  # Đọc từ Streamlit Secrets

# Thiết lập xác thực với DagsHub bằng Token
os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
os.environ["MLFLOW_TRACKING_TOKEN"] = mlflow_token

mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("MNIST_Classification")
client = MlflowClient()

st.title("MNIST Dimensionality Reduction with Streamlit & MLFlow")

# 1. Thu thập dữ liệu
st.header("📥 Thu thập dữ liệu")
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    return X, y

X, y = load_data()
st.success("✅ Dữ liệu MNIST đã được tải thành công!")

# 2. Chia dữ liệu
st.header("✂️ Chia dữ liệu")
test_size = st.slider("Chọn tỉ lệ tập kiểm tra:", 0.1, 0.5, 0.2, step=0.05)
if st.button("Xác nhận tỉ lệ và chia dữ liệu"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.session_state["X_train"] = X_train
    st.session_state["X_test"] = X_test
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.success(f"✅ Dữ liệu đã được chia: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra!")

# 3. Giảm chiều dữ liệu
st.header("📉 Giảm chiều dữ liệu")
reduction_method = st.selectbox("Chọn phương pháp giảm chiều:", ["PCA", "t-SNE"])
n_components = st.slider("Số chiều sau khi giảm:", 2, 50, 2)
if st.button("Thực hiện giảm chiều"):
    if "X_train" not in st.session_state:
        st.error("⚠️ Vui lòng chia dữ liệu trước!")
    else:
        X_sample = st.session_state["X_train"][:2000]
        y_sample = st.session_state["y_train"][:2000]
        with mlflow.start_run():
            if reduction_method == "PCA":
                reducer = PCA(n_components=n_components)
            else:
                reducer = TSNE(n_components=n_components, random_state=42)
            X_reduced = reducer.fit_transform(X_sample)
            mlflow.log_param("reduction_method", reduction_method)
            mlflow.log_param("n_components", n_components)
            st.success(f"✅ Giảm chiều dữ liệu thành công với {reduction_method}!")
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_sample, cmap='tab10', alpha=0.5)
            ax.legend(*scatter.legend_elements(), title="Nhóm")
            ax.set_title(f"Biểu đồ phân bố dữ liệu sau khi giảm chiều bằng {reduction_method}")
            st.pyplot(fig)