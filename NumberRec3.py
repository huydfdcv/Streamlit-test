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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
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

# Lý thuyết về K-Means và DBSCAN
def ly_thuyet_kmeans():
    st.subheader("📖 Lý thuyết về K-Means")
    st.write("""
    - **K-Means** là thuật toán phân cụm phổ biến nhất, hoạt động dựa trên nguyên tắc:
        1. Chọn ngẫu nhiên k điểm làm trung tâm cụm.
        2. Gán mỗi điểm dữ liệu vào cụm có trung tâm gần nhất.
        3. Cập nhật trung tâm cụm bằng cách tính trung bình các điểm trong cụm.
        4. Lặp lại cho đến khi không còn thay đổi.
    - K-Means hoạt động tốt với dữ liệu có cấu trúc rõ ràng nhưng nhạy cảm với outliers.
    """)
    st.image("kmeans.png", caption="Minh họa thuật toán K-Means")

def ly_thuyet_dbscan():
    st.subheader("📖 Lý thuyết về DBSCAN")
    st.write("""
    - **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là thuật toán phân cụm dựa trên mật độ:
        1. Xác định điểm lõi có ít nhất MinPts điểm lân cận trong phạm vi epsilon.
        2. Kết nối các điểm lõi với nhau để tạo thành cụm.
        3. Điểm nhiễu không thuộc cụm nào sẽ bị đánh dấu là outlier.
    - DBSCAN không cần xác định số cụm trước và hoạt động tốt với dữ liệu nhiễu.
    """)
    st.image("dbscan.png", caption="Minh họa thuật toán DBSCAN")

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
        
        mlflow.log_param("model", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, model_name)
        
        return model, acc

# Phân cụm K-Means và DBSCAN với PCA
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

st.title("MNIST Classification & Clustering with Streamlit & MLFlow")

# Tabs cho các chức năng
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Phân loại", "🔍 Phân cụm", "📖 K-Means", "📖 DBSCAN", "🎨 Dự đoán từ vẽ"])

with tab1:
    model_choice = st.selectbox("Chọn mô hình phân loại:", ["Decision Tree", "SVM"])
    if st.button("Huấn luyện"):
        model, acc = train_model(model_choice)
        st.success(f"Mô hình {model_choice} huấn luyện xong với độ chính xác: {acc:.4f}")

with tab2:
    clustering_choice = st.selectbox("Chọn thuật toán phân cụm:", ["K-Means", "DBSCAN"])
    if st.button("Thực hiện phân cụm"):
        clustering(clustering_choice)

with tab3:
    ly_thuyet_kmeans()

with tab4:
    ly_thuyet_dbscan()

with tab5:
    st.subheader("Vẽ số hoặc tải ảnh để dự đoán")
    
    # Kiểm tra xem thư viện có hoạt động đúng không
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
        
        if 'model' in locals():
            prediction = model.predict(image)[0]
            st.write(f"Dự đoán: {prediction}")
        else:
            st.error("Vui lòng huấn luyện mô hình trước!")