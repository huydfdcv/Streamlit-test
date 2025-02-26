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
from mlflow.tracking import MlflowClient
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Cấu hình MLflow
MLFLOW_TRACKING_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("MNIST_Classification")
client = MlflowClient()

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

# 2. Chia dữ liệu
st.header("✂️ Chia dữ liệu")
test_size = st.slider("Chọn tỉ lệ tập kiểm tra:", 0.1, 0.5, 0.2, step=0.05)
if st.button("Xác nhận tỉ lệ và chia dữ liệu"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
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

# 3. Tải dữ liệu từ MLflow
def load_data_from_mlflow():
    experiment = client.get_experiment_by_name("MNIST_Classification")
    runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
    if not runs:
        st.error("Không tìm thấy dữ liệu trên MLflow. Hãy chia dữ liệu trước.")
        return None, None, None, None
    run_id = runs[0].info.run_id
    client.download_artifacts(run_id, "X_train.npy", ".")
    client.download_artifacts(run_id, "X_test.npy", ".")
    client.download_artifacts(run_id, "y_train.npy", ".")
    client.download_artifacts(run_id, "y_test.npy", ".")
    return np.load("X_train.npy"), np.load("X_test.npy"), np.load("y_train.npy"), np.load("y_test.npy")

# 4. Chọn mô hình để huấn luyện
st.header("🎯 Chọn mô hình để huấn luyện")
def train_model(model_name):
    X_train, X_test, y_train, y_test = load_data_from_mlflow()
    if X_train is None:
        return None, None
    with mlflow.start_run():
        model = DecisionTreeClassifier() if model_name == "Decision Tree" else SVC()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        signature = infer_signature(X_test[:5], model.predict(X_test[:5]))
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, model_name, signature=signature, registered_model_name=model_name)
        return model, acc

def load_or_train_model(model_name):
    try:
        model_uri = f"models:/{model_name}/latest"
        return mlflow.sklearn.load_model(model_uri)
    except Exception:
        st.warning("🚀 Không tìm thấy mô hình trên MLflow. Bắt đầu huấn luyện...")
        return train_model(model_name)[0]

model_choice = st.selectbox("Chọn mô hình phân loại:", ["Decision Tree", "SVM"])
if st.button("Tải hoặc Huấn luyện mô hình"):
    model = load_or_train_model(model_choice)

# 5. Dự đoán & Đánh giá
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
    image = Image.open(canvas).convert('L') if canvas else Image.fromarray((255 - np.array(canvas_result.image_data[:, :, 0])).astype(np.uint8))
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, -1) / 255.0
    try:
        model_uri = f"models:/{model_choice}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        st.write(f"Dự đoán: {model.predict(image)[0]}")
    except Exception:
        st.error("Vui lòng huấn luyện mô hình trước!")
