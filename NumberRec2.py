import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
# Cấu hình MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"  # Thay bằng URL của MLflow server nếu chạy từ xa
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("MNIST_Classification")

st.title("MNIST Classification & Clustering with Streamlit & MLFlow")

# 1. Thu thập dữ liệu
st.header("📥 Thu thập dữ liệu")
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    return X, y

X, y = load_data()
st.success("✅ Dữ liệu MNIST đã được tải thành công!")

# 2. Chọn tỉ lệ tập kiểm tra và chia dữ liệu
st.header("✂️ Chia dữ liệu")
test_size = st.slider("Chọn tỉ lệ tập kiểm tra:", 0.1, 0.5, 0.2, step=0.05)
if st.button("Xác nhận tỉ lệ và chia dữ liệu"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Lưu từng tập dữ liệu lên MLflow
    with mlflow.start_run():
        np.save("X_train.npy", X_train)
        np.save("X_test.npy", X_test)
        np.save("y_train.npy", y_train)
        np.save("y_test.npy", y_test)
        mlflow.log_artifact("X_train.npy")
        mlflow.log_artifact("X_test.npy")
        mlflow.log_artifact("y_train.npy")
        mlflow.log_artifact("y_test.npy")
    
    st.success(f"✅ Dữ liệu đã được chia: {len(X_train)} mẫu huấn luyện, {len(X_test)} mẫu kiểm tra và lưu lên MLflow!")

# 3. Chọn mô hình để huấn luyện & logging
st.header("🎯 Chọn mô hình để huấn luyện")
def load_or_train_model(model_name):
    try:
        model_uri = f"models:/{model_name}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        st.info("🔄 Đã tải mô hình từ MLflow.")
    except Exception:
        st.warning("🚀 Không tìm thấy mô hình trên MLflow. Bắt đầu huấn luyện...")
        model, acc = train_model(model_name)
        mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
        st.success(f"Mô hình {model_name} đã được đăng ký trên MLflow với độ chính xác: {acc:.4f}")
    return model

def train_model(model_name):
    # Tải dữ liệu từ MLflow
    with mlflow.start_run():
        mlflow.artifacts.download_artifacts(artifact_path="X_train.npy", dst_path=".")
        mlflow.artifacts.download_artifacts(artifact_path="X_test.npy", dst_path=".")
        mlflow.artifacts.download_artifacts(artifact_path="y_train.npy", dst_path=".")
        mlflow.artifacts.download_artifacts(artifact_path="y_test.npy", dst_path=".")
        X_train = np.load("X_train.npy")
        X_test = np.load("X_test.npy")
        y_train = np.load("y_train.npy")
        y_test = np.load("y_test.npy")
    
    with mlflow.start_run():
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "SVM":
            model = SVC()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        input_example = X_test[:5]
        signature = infer_signature(input_example, model.predict(input_example))
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example, registered_model_name=model_name)
        
        return model, acc

model_choice = st.selectbox("Chọn mô hình phân loại:", ["Decision Tree", "SVM"])
if st.button("Tải hoặc Huấn luyện mô hình"):
    model = load_or_train_model(model_choice)

# 4. Dự đoán & Đánh giá
st.header("🔍 Dự đoán & Đánh giá")
st.subheader("Vẽ số hoặc tải ảnh để dự đoán")
try:
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True
    )
except Exception as e:
    st.error(f"Lỗi khi tải canvas: {e}")

canvas = st.file_uploader("Tải ảnh lên (28x28 px)", type=["png", "jpg", "jpeg"])

if canvas or (canvas_result is not None and canvas_result.image_data is not None):
    if canvas:
        image = Image.open(canvas).convert('L')
    elif canvas_result.image_data is not None:
        image = Image.fromarray((255 - np.array(canvas_result.image_data[:, :, 0])).astype(np.uint8))
    
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, -1) / 255.0
    
    try:
        model_uri = f"models:/{model_choice}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        prediction = model.predict(image)[0]
        st.write(f"Dự đoán: {prediction}")
    except Exception:
        st.error("Vui lòng huấn luyện mô hình trước!")


def clustering(algorithm):
    st.subheader(f"Phân cụm với {algorithm}")
    
    sample_size = 5000  # Giới hạn mẫu để tăng tốc độ
    X_sample = X_train[:sample_size]
    
    # Giảm chiều dữ liệu xuống 2D để hiển thị trực quan
    pca = PCA(n_components=2)
    X_sample_pca = pca.fit_transform(X_sample)
    
    if algorithm == "K-Means":
        model = KMeans(n_clusters=10, random_state=42)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=3, min_samples=10)
    
    clusters = model.fit_predict(X_sample_pca)
    
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_sample_pca[:, 0], X_sample_pca[:, 1], c=clusters, cmap='tab10', alpha=0.5)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cụm")
    ax.add_artist(legend1)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

clustering_choice = st.selectbox("Chọn thuật toán phân cụm:", ["K-Means", "DBSCAN"])
if st.button("Thực hiện phân cụm"):
    clustering(clustering_choice)