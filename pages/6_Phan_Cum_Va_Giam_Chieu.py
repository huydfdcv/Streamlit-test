import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import mlflow
import mlflow.sklearn
import os

# Thiết lập MLflow với DagsHub
DAGSHUB_MLFLOW_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
mlflow.set_experiment("MNIST Clustering")  # Đặt tên experiment

# Thiết lập thông tin xác thực DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2CaXhRNYabm9fN3"

# Tải dữ liệu MNIST
@st.cache_data
def load_data():
    """
    Tải tập dữ liệu MNIST từ OpenML.
    - MNIST bao gồm 70.000 ảnh chữ số viết tay (28x28 pixel).
    - Mỗi ảnh được biểu diễn dưới dạng vector 784 chiều (28x28 = 784).
    - `mnist.data` chứa các đặc trưng (pixel).
    - `mnist.target` chứa nhãn (chữ số từ 0 đến 9).
    """
    mnist = fetch_openml('mnist_784', version=1)
    return mnist.data, mnist.target

# Chuẩn hóa dữ liệu
def preprocess_data(X):
    """
    Chuẩn hóa dữ liệu bằng StandardScaler.
    - Đưa các giá trị pixel về cùng phạm vi (mean = 0, std = 1).
    - Giúp cải thiện hiệu suất của các thuật toán học máy.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Giảm chiều dữ liệu
def reduce_dimensions(X, n_components, method='PCA'):
    """
    Giảm chiều dữ liệu sử dụng PCA hoặc t-SNE.
    - PCA: Phân tích thành phần chính, giảm chiều dữ liệu dựa trên phương sai.
    - t-SNE: Giảm chiều dữ liệu dựa trên phân phối xác suất, phù hợp để trực quan hóa.
    """
    if method == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method == 't-SNE':
        reducer = TSNE(n_components=n_components, random_state=42)
    X_reduced = reducer.fit_transform(X)
    return X_reduced

# Phân cụm dữ liệu
def perform_clustering(X, method='K-means', n_clusters=10):
    """
    Phân cụm dữ liệu sử dụng K-means hoặc DBSCAN.
    - K-means: Phân cụm dữ liệu thành K cụm dựa trên khoảng cách Euclid.
    - DBSCAN: Phân cụm dữ liệu dựa trên mật độ, không cần chỉ định số cụm.
    """
    if method == 'K-means':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'DBSCAN':
        clusterer = DBSCAN(eps=0.5, min_samples=5)
    labels = clusterer.fit_predict(X)
    return clusterer, labels

# Hiển thị kết quả phân cụm
def display_clustering_results(X_reduced, labels, n_components):
    """
    Hiển thị kết quả phân cụm dưới dạng biểu đồ.
    - Nếu số chiều là 2: Hiển thị biểu đồ 2D.
    - Nếu số chiều là 3: Hiển thị biểu đồ 3D.
    - Nếu số chiều > 3: Thông báo không hiển thị trực quan.
    """
    st.write("### Kết quả phân cụm")
    
    if n_components == 2:
        st.write("#### Biểu đồ 2D")
        st.write("Biểu đồ 2D hiển thị các cụm dữ liệu sau khi giảm chiều.")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
        legend = ax.legend(*scatter.legend_elements(), title="Cụm")
        ax.add_artist(legend)
        st.pyplot(fig)
        
    elif n_components == 3:
        st.write("#### Biểu đồ 3D")
        st.write("Biểu đồ 3D tương tác hiển thị các cụm dữ liệu sau khi giảm chiều.")
        df = pd.DataFrame(X_reduced, columns=['x', 'y', 'z'])
        df['cluster'] = labels
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', title="Phân cụm 3D")
        st.plotly_chart(fig)
        
    else:
        st.write("Dữ liệu đã được giảm chiều và phân cụm, nhưng không hiển thị trực quan vì số chiều > 3.")

    # Tính toán và hiển thị silhouette score
    if len(np.unique(labels)) > 1:  # Chỉ tính silhouette score nếu có nhiều hơn 1 cụm
        silhouette_avg = silhouette_score(X_reduced, labels)
        st.write(f"#### Silhouette Score: {silhouette_avg:.2f}")
        st.write("Silhouette Score đo lường chất lượng phân cụm, giá trị càng gần 1 càng tốt.")
        return silhouette_avg
    return None

# Streamlit app
def main():
    st.title("Phân cụm dữ liệu MNIST với K-means và DBSCAN")
    st.write("""
    Ứng dụng này thực hiện phân cụm dữ liệu MNIST sử dụng hai phương pháp:
    - **K-means**: Phân cụm dựa trên khoảng cách.
    - **DBSCAN**: Phân cụm dựa trên mật độ.
    """)

    # Tải dữ liệu
    st.write("## Bước 1: Tải và chuẩn hóa dữ liệu")
    st.write("Tập dữ liệu MNIST bao gồm 70.000 ảnh chữ số viết tay (28x28 pixel).")
    X, y = load_data()
    st.write(f"- Số lượng mẫu: {X.shape[0]}")
    st.write(f"- Số lượng đặc trưng: {X.shape[1]}")

    # Chuẩn hóa dữ liệu
    st.write("### Chuẩn hóa dữ liệu")
    st.write("Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng phạm vi giá trị.")
    X_scaled = preprocess_data(X)

    # Chọn phương pháp giảm chiều
    st.write("## Bước 2: Giảm chiều dữ liệu")
    st.write("""
    Giảm chiều dữ liệu giúp giảm độ phức tạp và dễ dàng trực quan hóa.
    - **PCA**: Phù hợp để giảm chiều dữ liệu với số lượng lớn.
    - **t-SNE**: Phù hợp để trực quan hóa dữ liệu.
    """)
    reduction_method = st.selectbox("Phương pháp giảm chiều", ["PCA", "t-SNE"])

    # Chọn số chiều
    st.write("### Chọn số chiều")
    st.write("Chọn số chiều để giảm dữ liệu. Nếu chọn 2 hoặc 3, dữ liệu sẽ được hiển thị trên biểu đồ.")
    n_components = st.slider("Số chiều", 2, 50, 2)

    # Nút xác nhận giảm chiều
    if st.button("Xác nhận giảm chiều"):
        st.session_state['X_reduced'] = reduce_dimensions(X_scaled, n_components, method=reduction_method)
        st.write(f"Dữ liệu đã được giảm chiều từ {X.shape[1]} chiều xuống {n_components} chiều sử dụng {reduction_method}.")

    # Kiểm tra nếu dữ liệu đã được giảm chiều
    if 'X_reduced' in st.session_state:
        X_reduced = st.session_state['X_reduced']

        # Chọn phương pháp phân cụm
        st.write("## Bước 3: Phân cụm dữ liệu")
        st.write("""
        Chọn phương pháp phân cụm:
        - **K-means**: Cần chỉ định số cụm (K).
        - **DBSCAN**: Không cần chỉ định số cụm, tự động xác định dựa trên mật độ.
        """)
        clustering_method = st.selectbox("Phương pháp phân cụm", ["K-means", "DBSCAN"])

        # Phân cụm dữ liệu
        if clustering_method == "K-means":
            n_clusters = st.slider("Chọn số cụm (K)", 2, 20, 10)
        else:
            n_clusters = None  # DBSCAN không cần số cụm

        if st.button("Thực hiện phân cụm"):
            with mlflow.start_run():
                # Log các tham số
                mlflow.log_param("reduction_method", reduction_method)
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("clustering_method", clustering_method)
                if clustering_method == "K-means":
                    mlflow.log_param("n_clusters", n_clusters)

                # Phân cụm dữ liệu
                clusterer, labels = perform_clustering(X_reduced, method=clustering_method, n_clusters=n_clusters)
                
                # Hiển thị kết quả phân cụm
                silhouette_avg = display_clustering_results(X_reduced, labels, n_components)

                # Log silhouette score
                if silhouette_avg is not None:
                    mlflow.log_metric("silhouette_score", silhouette_avg)

                # Log mô hình
                if clustering_method == "K-means":
                    mlflow.sklearn.log_model(clusterer, "kmeans_model")
                elif clustering_method == "DBSCAN":
                    mlflow.sklearn.log_model(clusterer, "dbscan_model")

                st.write("Kết quả đã được log lên DagsHub MLflow!")
                st.markdown(f"[Xem thí nghiệm trên DagsHub]({DAGSHUB_MLFLOW_URI})")

if __name__ == "__main__":
    main()
