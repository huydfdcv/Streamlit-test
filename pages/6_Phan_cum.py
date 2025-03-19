import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, confusion_matrix
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
def load_data(n_samples):
    """
    Tải tập dữ liệu MNIST từ OpenML.
    - MNIST bao gồm 70.000 ảnh chữ số viết tay (28x28 pixel).
    - Mỗi ảnh được biểu diễn dưới dạng vector 784 chiều (28x28 = 784).
    - mnist.data chứa các đặc trưng (pixel).
    - mnist.target chứa nhãn (chữ số từ 0 đến 9).
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data[:n_samples]
    y = mnist.target[:n_samples]
    
    # Chuẩn hóa nhãn về dạng số (nếu nhãn là chuỗi)
    y = y.astype(int)
    
    return X, y

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
def reduce_dimensions(X, n_components):
    """
    Giảm chiều dữ liệu sử dụng PCA (nhanh hơn t-SNE).
    - PCA: Phân tích thành phần chính, giảm chiều dữ liệu dựa trên phương sai.
    """
    reducer = PCA(n_components=n_components, random_state=42)
    X_reduced = reducer.fit_transform(X)
    return X_reduced

# Phân cụm dữ liệu
def perform_clustering(X, method='K-means', n_clusters=10, **kwargs):
    """
    Phân cụm dữ liệu sử dụng K-means hoặc DBSCAN.
    - K-means: Phân cụm dữ liệu thành K cụm dựa trên khoảng cách Euclid.
    - DBSCAN: Phân cụm dữ liệu dựa trên mật độ, không cần chỉ định số cụm.
    """
    if method == 'K-means':
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
    elif method == 'DBSCAN':
        clusterer = DBSCAN(**kwargs)
    labels = clusterer.fit_predict(X)
    return clusterer, labels

# Hiển thị kết quả phân cụm
def display_clustering_results(X_reduced, labels, true_labels, n_components):
    """
    Hiển thị kết quả phân cụm dưới dạng biểu đồ.
    - Nếu số chiều là 2: Hiển thị biểu đồ 2D.
    - Nếu số chiều là 3: Hiển thị biểu đồ 3D.
    - Mỗi cụm được hiển thị với màu riêng biệt và nhãn tương ứng.
    """
    st.write("### Kết quả phân cụm")
    
    # Tạo bảng ánh xạ giữa cụm và nhãn thực tế
    confusion_mat = confusion_matrix(true_labels, labels)
    cluster_to_label = np.argmax(confusion_mat, axis=0)
    
    # Đổi nhãn cụm thành nhãn thực tế
    mapped_labels = np.array([cluster_to_label[label] for label in labels])
    
    # Tạo bảng màu riêng biệt cho các cụm
    colors = plt.cm.tab20(np.linspace(0, 1, len(np.unique(labels))))
    
    if n_components == 2:
        st.write("#### Biểu đồ 2D")
        fig, ax = plt.subplots()
        for i, color in zip(np.unique(labels), colors):
            ax.scatter(X_reduced[labels == i, 0], X_reduced[labels == i, 1], 
                       color=color, label=f'Cụm {i} (Số {cluster_to_label[i]})', alpha=0.6)
        ax.legend(title="Cụm và nhãn thực tế")
        st.pyplot(fig)
        
    elif n_components == 3:
        st.write("#### Biểu đồ 3D")
        df = pd.DataFrame(X_reduced, columns=['x', 'y', 'z'])
        df['cluster'] = mapped_labels
        df['true_label'] = true_labels
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='cluster', 
                            title="Phân cụm 3D", color_continuous_scale=px.colors.qualitative.Set1)
        st.plotly_chart(fig)
        
    # Tính toán và hiển thị silhouette score
    if len(np.unique(labels)) > 1:  # Chỉ tính silhouette score nếu có nhiều hơn 1 cụm
        silhouette_avg = silhouette_score(X_reduced, labels)
        st.write(f"#### Silhouette Score: {silhouette_avg:.2f}")
        st.write("Silhouette Score đo lường chất lượng phân cụm, giá trị càng gần 1 càng tốt.")
        return silhouette_avg
    return None

# Hiển thị một mẫu từ mỗi cụm
def display_samples_from_clusters(X, labels, true_labels):
    """
    Hiển thị một mẫu từ mỗi cụm để kiểm tra độ chính xác của phân cụm.
    """
    st.write("### Mẫu từ mỗi cụm")
    unique_clusters = np.unique(labels)
    fig, axes = plt.subplots(1, len(unique_clusters), figsize=(15, 3))
    for i, cluster in enumerate(unique_clusters):
        sample_index = np.where(labels == cluster)[0][0]  # Lấy mẫu đầu tiên từ cụm
        sample_image = X[sample_index].reshape(28, 28)
        axes[i].imshow(sample_image, cmap='gray')
        axes[i].set_title(f"Cụm {cluster} (Số {true_labels[sample_index]})")
        axes[i].axis('off')
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Phân cụm dữ liệu MNIST với K-means và DBSCAN")
    st.write("""
    Ứng dụng này thực hiện phân cụm dữ liệu MNIST sử dụng hai phương pháp:
    - **K-means**: Phân cụm dựa trên khoảng cách.
    - **DBSCAN**: Phân cụm dựa trên mật độ.
    """)

    # Khởi tạo session state
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    if 'clustering_done' not in st.session_state:
        st.session_state['clustering_done'] = False
    if 'reduction_done' not in st.session_state:
        st.session_state['reduction_done'] = False

    # Chọn số mẫu
    st.write("## Bước 1: Chọn số mẫu")
    n_samples = st.slider("Chọn số mẫu", 10000, 70000, 10000)

    # Nút xác nhận chia dữ liệu
    if st.button("Xác nhận chia dữ liệu"):
        # Tải dữ liệu
        st.write("## Bước 2: Tải và chuẩn hóa dữ liệu")
        st.write(f"Tập dữ liệu MNIST bao gồm {n_samples} ảnh chữ số viết tay (28x28 pixel).")
        X, y = load_data(n_samples)
        st.write(f"- Số lượng mẫu: {X.shape[0]}")
        st.write(f"- Số lượng đặc trưng: {X.shape[1]}")

        # Chuẩn hóa dữ liệu
        st.write("### Chuẩn hóa dữ liệu")
        st.write("Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng phạm vi giá trị.")
        X_scaled = preprocess_data(X)

        # Lưu dữ liệu vào session state
        st.session_state['X_scaled'] = X_scaled
        st.session_state['y'] = y
        st.session_state['data_loaded'] = True

    # Kiểm tra nếu dữ liệu đã được tải
    if st.session_state['data_loaded']:
        X_scaled = st.session_state['X_scaled']
        y = st.session_state['y']

        # Chọn phương pháp phân cụm
        st.write("## Bước 3: Phân cụm dữ liệu")
        st.write("""
        Chọn phương pháp phân cụm:
        - **K-means**: Phân cụm dựa trên khoảng cách.
        - **DBSCAN**: Phân cụm dựa trên mật độ.
        """)
        clustering_method = st.selectbox("Phương pháp phân cụm", ["K-means", "DBSCAN"])

        # Tùy chỉnh thông số thuật toán
        st.write("### Tùy chỉnh thông số thuật toán")
        if clustering_method == "K-means":
            init = st.selectbox("Phương pháp khởi tạo", ["k-means++", "random"])
            max_iter = st.slider("Số lần lặp tối đa", 100, 1000, 300)
            random_state = st.number_input("Random state", value=42)
            kwargs = {"init": init, "max_iter": max_iter, "random_state": random_state}
        elif clustering_method == "DBSCAN":
            eps = st.slider("Khoảng cách tối đa (eps)", 0.1, 1.0, 0.5)
            min_samples = st.slider("Số mẫu tối thiểu", 1, 20, 5)
            kwargs = {"eps": eps, "min_samples": min_samples}

        # Phân cụm dữ liệu
        if st.button("Thực hiện phân cụm"):
            with mlflow.start_run():
                # Log các tham số
                mlflow.log_param("n_samples", n_samples)
                mlflow.log_param("clustering_method", clustering_method)
                mlflow.log_params(kwargs)

                # Phân cụm dữ liệu
                clusterer, labels = perform_clustering(X_scaled, method=clustering_method, n_clusters=10, **kwargs)
                
                # Lưu kết quả vào session state
                st.session_state['clusterer'] = clusterer
                st.session_state['labels'] = labels
                st.session_state['clustering_done'] = True

                st.write("Phân cụm đã hoàn thành!")

        # Giảm chiều dữ liệu
        if st.session_state['clustering_done']:
            st.write("## Bước 4: Giảm chiều dữ liệu")
            st.write("Giảm chiều dữ liệu để trực quan hóa kết quả phân cụm.")
            n_components = st.selectbox("Số chiều", [2, 3])
            if st.button("Xác nhận giảm chiều"):
                X_reduced = reduce_dimensions(X_scaled, n_components)
                
                # Lưu kết quả vào session state
                st.session_state['X_reduced'] = X_reduced
                st.session_state['reduction_done'] = True

        # Hiển thị kết quả
        if st.session_state['reduction_done']:
            X_reduced = st.session_state['X_reduced']
            labels = st.session_state['labels']
            
            # Hiển thị kết quả phân cụm
            silhouette_avg = display_clustering_results(X_reduced, labels, y, n_components)

            # Hiển thị một mẫu từ mỗi cụm
            display_samples_from_clusters(X_scaled, labels, y)

            # Log silhouette score
            if silhouette_avg is not None:
                mlflow.log_metric("silhouette_score", silhouette_avg)

            # Log mô hình
            if clustering_method == "K-means":
                mlflow.sklearn.log_model(st.session_state['clusterer'], "kmeans_model")
            elif clustering_method == "DBSCAN":
                mlflow.sklearn.log_model(st.session_state['clusterer'], "dbscan_model")

            st.write("Kết quả đã được log lên DagsHub MLflow!")
            st.markdown(f"[Xem thí nghiệm trên DagsHub]({DAGSHUB_MLFLOW_URI})")

if __name__ == "__main__":
    main()