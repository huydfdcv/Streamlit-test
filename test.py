import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ========== 🚀 Tải dữ liệu ==========
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# ========== 📌 Tạo Tabs ==========
tabs = st.tabs(["🏠 Giới thiệu", "📊 Xử lý dữ liệu", "🎯 Huấn luyện", "🔮 Dự đoán", "🔍 MLflow Tracking"])

# ========== 🏠 TAB 1: GIỚI THIỆU ==========
with tabs[0]:
    st.title("🚢 Titanic Survival Prediction")
    st.write("🔍 Dự đoán khả năng sống sót của hành khách Titanic bằng hồi quy.")

# ========== 📊 TAB 2: XỬ LÝ DỮ LIỆU ==========
with tabs[1]:
    st.subheader("📊 Xử lý dữ liệu")
    
    # Hiển thị dữ liệu gốc
    st.write("📌 **10 dòng đầu tiên của dữ liệu:**")
    st.write(df.head())

    # Chọn cột để xóa
    df.drop(columns=['Name', 'PassengerId','Ticket', 'Cabin'], inplace=True)
    drop_columns = st.multiselect("📌 **Chọn cột để xóa**", ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'])
    
    # Xử lý dữ liệu
    df_cleaned = df.copy() 
    df_cleaned.drop(columns=drop_columns, inplace=True)

    # Điền giá trị thiếu
    if 'Age' not in drop_columns:
        df_cleaned['Age'].fillna(df_cleaned['Age'].median(), inplace=True)
    if 'Embarked' not in drop_columns:
        df_cleaned['Embarked'].fillna(df_cleaned['Embarked'].mode()[0], inplace=True)

    # One-hot encoding
    if 'Sex' not in drop_columns:
        df_cleaned = pd.get_dummies(df_cleaned, columns=['Sex'], drop_first=True)
    if 'Embarked' not in drop_columns:
        df_cleaned = pd.get_dummies(df_cleaned, columns=['Embarked'], drop_first=False)

    # Hiển thị dữ liệu sau khi xử lý
    st.write("📌 **Dữ liệu sau khi xử lý:**")
    st.write(df_cleaned.head())

# ========== 🎯 TAB 3: HUẤN LUYỆN MÔ HÌNH ==========
with tabs[2]:
    st.subheader("🎯 Huấn luyện mô hình")

    # Chia dữ liệu
    X = df_cleaned.drop(columns=['Survived'])
    y = df_cleaned['Survived']
    
    test_size = st.slider("📌 Chọn tỷ lệ test:", 0.1, 0.5, 0.2)
    valid_size = st.slider("📌 Chọn tỷ lệ validation:", 0.1, 0.5, 0.2)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # Chọn mô hình
    model_type = st.radio("📌 **Chọn mô hình:**", ["Linear Regression", "Polynomial Regression"])
    degree = st.slider("📌 Bậc của Polynomial Regression:", 2, 5, 2) if model_type == "Polynomial Regression" else 1

    def train_model():
        mlflow.set_experiment("Titanic_Regression")
        with mlflow.start_run():
            if model_type == "Polynomial Regression":
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train_scaled)
                X_valid_poly = poly.transform(X_valid_scaled)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                valid_pred = model.predict(X_valid_poly)
            else:
                model = LinearRegression()
                model.fit(X_train_scaled, y_train)
                valid_pred = model.predict(X_valid_scaled)

            valid_mse = mean_squared_error(y_valid, valid_pred)
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("validation_mse", valid_mse)
            mlflow.sklearn.log_model(model, "regression_model")

            return model, valid_mse

    if st.button("🚀 Huấn luyện mô hình"):
        model, valid_mse = train_model()
        st.success(f"✅ Huấn luyện thành công! MSE trên tập Validation: {valid_mse:.4f}")

# ========== 🔮 TAB 4: DỰ ĐOÁN ==========
with tabs[3]:
    st.subheader("🔮 Dự đoán sống sót trên Titanic")

    # Nhập dữ liệu
    input_data = {
        'Pclass': st.selectbox("📌 Hạng vé", [1, 2, 3]),
        'Age': st.slider("📌 Tuổi", 1, 100, 30),
        'SibSp': st.slider("📌 Số anh chị em / vợ chồng", 0, 8, 0),
        'Parch': st.slider("📌 Số cha mẹ / con", 0, 6, 0),
        'Fare': st.slider("📌 Giá vé", 0, 500, 50),
        'Sex_male': 1 if st.radio("📌 Giới tính", ["Nam", "Nữ"]) == "Nam" else 0,
        'Embarked_C': 0, 'Embarked_Q': 0, 'Embarked_S': 0
    }
    
    # Xử lý nơi lên tàu
    embarked = st.radio("📌 Nơi lên tàu", ["C", "Q", "S"])
    input_data[f'Embarked_{embarked}'] = 1

    # Chuyển đổi thành DataFrame
    input_df = pd.DataFrame([input_data])

    # Chuẩn hóa dữ liệu
    input_scaled = scaler.transform(input_df)

    if st.button("🔮 Dự đoán"):
        model, _ = train_model()
        prediction = model.predict(input_scaled)
        st.write(f"🔮 **Xác suất sống sót: {prediction[0]:.2f}**")
        st.success("✅ Sống sót!") if prediction[0] > 0.5 else st.error("❌ Không sống sót.")

# ========== 🔍 TAB 5: MLflow Tracking ==========
with tabs[4]:
    st.subheader("🔍 MLflow Tracking")
    st.markdown("👉 **Nhấn vào đây để xem chi tiết:**")
    st.link_button(label="📌 Mở MLflow", url="http://127.0.0.1:5000/#/experiments/")
