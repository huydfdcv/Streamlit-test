import streamlit as st
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps
import cv2
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
        mlflow.set_tracking_uri('http://localhost:5000')
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_name == "SVM":
            model = SVC()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, model_name)
        
        return model, acc

st.title("MNIST Classification with Streamlit & MLFlow")
model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])
if st.button("Huấn luyện"):
    model, acc = train_model(model_choice)
    st.success(f"Mô hình {model_choice} huấn luyện xong với độ chính xác: {acc:.4f}")


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

if (canvas_result is not None and canvas_result.image_data is not None):
    image = Image.fromarray((255 - canvas_result.image_data[:, :, 0]).astype(np.uint8))  
    image = image.resize((28, 28))
    image = np.array(image).reshape(1, -1) / 255.0
    
    if 'model' in locals():
        prediction = model.predict(image)[0]
        st.write(f"Dự đoán: {prediction}")
    else:
        st.error("Vui lòng huấn luyện mô hình trước!")

