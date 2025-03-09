import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import os
from PIL import Image

# Thiết lập MLflow tracking URI và thông tin xác thực
DAGSHUB_MLFLOW_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
mlflow.set_experiment("MNIST Classification")  # Đặt tên experiment

os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2CaXhRNYabm9fN3"

# Load dataset MNIST
@st.cache_data
def load_data():
    mnist = fetch_openml('mnist_784', version=1)
    return mnist.data, mnist.target

# Preprocess data
@st.cache_data
def preprocess_data(X, y):
    X = X / 255.0  # Chuẩn hóa dữ liệu về [0, 1]
    y = y.astype(int)  # Chuyển đổi nhãn thành số nguyên
    return X, y

# Split data
def split_data(X, y, test_size):
    st.write(f"Chia dữ liệu thành tập huấn luyện và tập kiểm tra với tỉ lệ test size = {test_size}.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Hiển thị báo cáo phân loại dưới dạng bảng
def display_classification_report(y_test, y_pred):
    """
    Hiển thị báo cáo phân loại dưới dạng bảng.
    """
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("### Báo cáo phân loại")
    st.dataframe(report_df.style.format("{:.2f}"))  # Hiển thị bảng với định dạng số thập phân

# Train and evaluate model
def train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name):
    with mlflow.start_run():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Độ chính xác của mô hình {model_name}: {accuracy:.2f}")
        
        # Hiển thị báo cáo phân loại dưới dạng bảng
        display_classification_report(y_test, y_pred)
        
        # Log metrics and model
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, model_name)
        st.session_state['model'] = model  # Lưu model vào session state

# Streamlit app
def main():
    st.title("MNIST Classification with Streamlit & MLflow (DagsHub)")
    
    # Nút link đến DagsHub
    st.markdown(f"[View logged experiments on DagsHub]({DAGSHUB_MLFLOW_URI})")

    # Bước 1: Thu thập dữ liệu
    st.write("""
    Tải tập dữ liệu MNIST từ OpenML.
    - MNIST bao gồm 70.000 ảnh chữ số viết tay (28x28 pixel).
    - Mỗi ảnh được biểu diễn dưới dạng vector 784 chiều (28x28 = 784).
    Gồm:
    - `mnist.data` chứa các đặc trưng (pixel).
    - `mnist.target` chứa nhãn (chữ số từ 0 đến 9).
    """)
    if st.button("1. Thu thập dữ liệu"):
        st.session_state['data_loaded'] = True
        st.write("Dữ liệu đã được tải thành công!")

    if st.session_state.get('data_loaded', False):
        X, y = load_data()
        X, y = preprocess_data(X, y)

        # Bước 2: Xử lý dữ liệu
        if st.button("2. Xử lý dữ liệu"):
            st.write("""
            Xử lý dữ liệu:
            - Chuẩn hóa giá trị pixel về khoảng [0, 1] bằng cách chia cho 255.
            - Chuyển đổi nhãn thành số nguyên.
            """)
            st.session_state['data_preprocessed'] = True
            st.write("Dữ liệu đã được xử lý (chuẩn hóa).")

        if st.session_state.get('data_preprocessed', False):
            # Bước 3: Chia dữ liệu
            st.write(
            """
            Chia dữ liệu thành tập huấn luyện và tập kiểm tra.
            - Tỉ lệ dữ liệu dùng để kiểm tra (ví dụ: 0.2 nghĩa là 20% dữ liệu dùng để kiểm tra).
            """
            )
            test_size = st.slider("Chọn tỉ lệ test/train", 0.1, 0.5, 0.2)
            if st.button("3. Chia dữ liệu"):
                X_train, X_test, y_train, y_test = split_data(X,y, test_size)
                st.write(f"Dữ liệu đã được chia (test size = {test_size}).")

            if X_train is not None:
                # Bước 4: Chọn mô hình và huấn luyện
                st.write("""
                    Huấn luyện và đánh giá mô hình:
                    - Huấn luyện mô hình trên tập huấn luyện.
                    - Đánh giá mô hình trên tập kiểm tra.
                    - Log độ chính xác và mô hình lên MLflow.
                    """)
                st.write("### 4. Chọn mô hình và huấn luyện")
                model_name = st.selectbox("Chọn mô hình", ["Decision Tree", "SVM"])
                if st.button("Huấn luyện mô hình"):
                    if model_name == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)
                    elif model_name == "SVM":
                        model = SVC(kernel='linear', random_state=42)
                    train_and_evaluate(model,X_train,X_test,y_train,y_test, model_name)
                    st.write("Mô hình đã được huấn luyện và log lên MLflow!")

                # Bước 5: Demo với ảnh tải lên
                st.write("### 5. Demo với ảnh tải lên")
                uploaded_file = st.file_uploader("Tải lên ảnh chữ số (28x28 pixels)", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    image = Image.open(uploaded_file).convert('L')  # Chuyển sang ảnh xám
                    image = image.resize((28, 28))
                    
                    # Chuyển ảnh thành mảng numpy
                    img_array = np.array(image).reshape(1, -1) / 255.0
                    
                    if st.button("Dự đoán"):
                        if 'model' in st.session_state:
                            prediction = st.session_state['model'].predict(img_array)
                            st.write(f"Kết quả dự đoán: {prediction[0]}")
                        else:
                            st.write("Vui lòng huấn luyện mô hình trước khi dự đoán.")

if __name__ == "__main__":
    main()
