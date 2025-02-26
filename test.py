import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import tempfile
import re

# Khởi tạo MLflow
mlflow.set_experiment("Titanic_prediction")
mlflow.set_tracking_uri("https://dagshub.com/huydfdcv/my-first-repo.mlflow")
os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c7c6bddfd4cca54d0c0b6fb70c7e45af45b22d91"

# ---- 🎯 Cài đặt giao diện ----
st.set_page_config(page_title="Titanic Survival Prediction", layout="wide")
st.title("🚢 Titanic Survival Prediction")
st.write("Dự đoán khả năng sống sót của hành khách trên Titanic bằng Hồi quy tuyến tính.")

# ---- 📌 Load dữ liệu ----
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df.copy()

if "df" not in st.session_state:
    st.session_state["df"] = load_data()

# ---- 🔄 Tiền xử lý dữ liệu ----
st.subheader("1️⃣ Tiền xử lý dữ liệu")
df = st.session_state["df"].copy()
st.write("📌 **Dữ liệu gốc**:")
st.write(df.head())

drop_columns = st.multiselect("🔧 Chọn các cột để xóa", df.columns.tolist())

if st.button("✅ Xác nhận xóa cột"):
    df.drop(columns=drop_columns, inplace=True)
    st.session_state["df"] = df.copy()
    st.success(f"Đã xóa các cột: {', '.join(drop_columns)}")

# Xử lý cột Ticket: chỉ giữ lại số
if 'Ticket' in df.columns:
    df['Ticket'] = df['Ticket'].apply(lambda x: re.sub(r'\D', '', str(x)))
    df['Ticket'] = pd.to_numeric(df['Ticket'], errors='coerce').fillna(0).astype(int)

# Điền giá trị thiếu
if 'Age' in df.columns:
    df['Age'].fillna(df['Age'].median(), inplace=True)
if 'Embarked' in df.columns:
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# One-Hot Encoding
if 'Sex' in df.columns:
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
if 'Embarked' in df.columns:
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

st.session_state["df"] = df.copy()
st.write("📌 **Dữ liệu sau xử lý**:")
st.write(df.head())

# ---- 📊 Chia dữ liệu ----
st.subheader("2️⃣ Chia dữ liệu")
if "df" in st.session_state:
    df = st.session_state["df"].copy()
else:
    st.error("⚠️ Vui lòng thực hiện tiền xử lý dữ liệu trước!")
    st.stop()

X = df.drop(columns=['Survived'])
y = df['Survived']

test_size = st.slider("📏 Chọn tỷ lệ tập test:", 0.1, 0.5, 0.2)
valid_size = st.slider("📏 Chọn tỷ lệ tập validation:", 0.1, 0.5, 0.2)

if st.button("🔀 Chia dữ liệu"):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)
    st.session_state["X_train"], st.session_state["X_valid"], st.session_state["y_train"], st.session_state["y_valid"] = X_train, X_valid, y_train, y_valid
    st.success("✅ Dữ liệu đã được chia thành công!")

# ---- 📉 Huấn luyện mô hình ----
st.subheader("3️⃣ Huấn luyện mô hình")
if "X_train" not in st.session_state:
    st.error("⚠️ Vui lòng chia dữ liệu trước!")
    st.stop()

X_train, X_valid, y_train, y_valid = st.session_state["X_train"], st.session_state["X_valid"], st.session_state["y_train"], st.session_state["y_valid"]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

model_type = st.radio("📌 Chọn mô hình:", ["Linear Regression", "Polynomial Regression"])
if st.button("✅ Xác nhận mô hình"):
    st.session_state["model_type"] = model_type
    st.success(f"Mô hình đã chọn: {model_type}")

if "model_type" in st.session_state:
    degree = 2
    if st.session_state["model_type"] == "Polynomial Regression":
        degree = st.slider("🔢 Chọn bậc của Polynomial Regression:", 2, 5, 2)
    
    with mlflow.start_run():
        if st.session_state["model_type"] == "Polynomial Regression":
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_valid_poly = poly.transform(X_valid_scaled)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
        else:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
        
        mlflow.sklearn.log_model(model, artifact_path="models")
        st.success("✅ Huấn luyện mô hình thành công!")

# ---- 🔮 Dự đoán ----
st.subheader("4️⃣ Dự đoán")
model_uri = "models:/regression_model/latest"
model = mlflow.sklearn.load_model(model_uri)

st.write("📥 Nhập dữ liệu hành khách:")
input_data = {col: st.number_input(f"{col}", value=0.0) for col in X.columns}
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)

st.write(f"🔮 **Dự đoán xác suất sống sót: {prediction[0]:.4f}**")
