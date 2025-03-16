import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load MNIST dataset
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_data()

# Huấn luyện mô hình
def train_model(model_name):
    with mlflow.start_run():
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "SVM":
            model = SVC()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        # Tạo input_example và signature để MLflow nhận diện đầu vào
        input_example = X_test[:5]
        signature = infer_signature(input_example, model.predict(input_example))
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=input_example)
        
        return model, acc


st.title("MNIST Classification with Streamlit & MLFlow")
st.write("1 snipet từ tập dữ liệu mnist")
st.image("img3.png")
st.write("Mỗi ảnh đây điều là 1 số viết tay với kích cỡ 28 x 28 px")
model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
if st.button("Huấn luyện"):
    model, acc = train_model(model_choice)
    st.success(f"Mô hình {model_choice} huấn luyện xong với độ chính xác: {acc:.4f}")
    st.session_state["model"] = model

model = st.session_state['model']
st.subheader("Vẽ số hoặc tải ảnh để dự đoán")
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)
canvas = st.file_uploader("Tải ảnh lên (28x28 px)", type=["png", "jpg", "jpeg"])
if canvas or (canvas_result is not None and canvas_result.image_data is not None):
        if canvas:
            image = Image.open(canvas).convert('L')
        elif canvas_result.image_data is not None:
            image = Image.fromarray((255 - np.array(canvas_result.image_data[:, :, 0])).astype(np.uint8))
        
        image = image.resize((28, 28))
        image = np.array(image).reshape(1, -1) / 255.0
        
        if 'model' in st.session_state:

            prediction = model.predict(image)[0]
            st.write(f"Dự đoán: {prediction}")
        else:
            st.error("Vui lòng huấn luyện mô hình trước!")

