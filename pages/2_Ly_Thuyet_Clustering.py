import streamlit as st

# Hiển thị thông tin về K-means và DBSCAN
def display_clustering_info():
    st.title("Giới thiệu về K-means và DBSCAN")

    st.write("""
    ## K-means
    **K-means** là một thuật toán phân cụm (clustering) phổ biến, được sử dụng để chia dữ liệu thành K cụm dựa trên khoảng cách giữa các điểm dữ liệu.
    """)

    st.write("### Cách hoạt động")
    st.write("""
    - **Bước 1**: Chọn K điểm làm tâm cụm ban đầu (centroids).
    - **Bước 2**: Gán mỗi điểm dữ liệu vào cụm có centroid gần nhất.
    - **Bước 3**: Cập nhật vị trí của các centroids bằng cách tính trung bình của tất cả các điểm trong cụm.
    - **Bước 4**: Lặp lại Bước 2 và Bước 3 cho đến khi các centroids không thay đổi hoặc đạt số lần lặp tối đa.
    """)

    st.write("### Công thức toán học của K-Means")
    st.latex(r"""
    1. \text{Hàm khoảng cách: } d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
    """)
    st.latex(r"""
    2. \text{Cập nhật cụm: } C_j = \{ x_i \ | \ \arg\min_{j} d(x_i, \mu_j) \}
    """)
    st.latex(r"""
    3. \text{Cập nhật tâm cụm: } \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
    """)
    st.latex(r"""
    4. \text{Hàm mục tiêu: } J = \sum_{j=1}^{K} \sum_{x_i \in C_j} ||x_i - \mu_j||^2
    """)

    st.write("### Ưu điểm")
    st.write("""
    - Đơn giản và dễ hiểu.
    - Hiệu quả với dữ liệu có kích thước lớn.
    - Dễ dàng triển khai và mở rộng.
    """)

    st.write("### Nhược điểm")
    st.write("""
    - Cần chỉ định số cụm K trước.
    - Nhạy cảm với vị trí khởi tạo centroids.
    - Không hiệu quả với dữ liệu có hình dạng phức tạp hoặc nhiễu.
    """)

    st.write("### Ứng dụng")
    st.write("""
    - Phân cụm khách hàng trong marketing.
    - Phân loại tài liệu.
    - Phân tích hình ảnh.
    """)

    st.write("---")

    st.write("""
    ## DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    **DBSCAN** là một thuật toán phân cụm dựa trên mật độ, có khả năng phát hiện các cụm có hình dạng bất kỳ và loại bỏ nhiễu.
    """)

    st.write("### Cách hoạt động")
    st.write("""
    - **Bước 1**: Chọn một điểm dữ liệu ngẫu nhiên.
    - **Bước 2**: Tìm tất cả các điểm trong bán kính `eps` (epsilon) xung quanh điểm đó.
    - **Bước 3**: Nếu số lượng điểm trong bán kính `eps` đạt ngưỡng `min_samples`, điểm đó được coi là điểm lõi (core point) và tạo thành một cụm.
    - **Bước 4**: Lặp lại quá trình với các điểm lân cận của điểm lõi để mở rộng cụm.
    - **Bước 5**: Các điểm không thuộc bất kỳ cụm nào được coi là nhiễu (noise).
    """)

    st.write("### Công thức toán học của DBSCAN")
    st.latex(r"""
    - \text{Tham số: } \varepsilon \text{ (epsilon), } MinPts
    """)
    st.latex(r"""
    - \text{Điểm core: } N_\varepsilon(p) \geq MinPts
    """)
    st.latex(r"""
    - \text{Điểm border: } N_\varepsilon(p) < MinPts \text{ nhưng thuộc } \varepsilon \text{-lân cận của một điểm core}
    """)
    st.latex(r"""
    - \text{Điểm outlier: } \text{Không thuộc bất kỳ cụm nào}
    """)

    st.write("### Ưu điểm")
    st.write("""
    - Không cần chỉ định số cụm trước.
    - Có thể phát hiện các cụm có hình dạng bất kỳ.
    - Loại bỏ được nhiễu trong dữ liệu.
    """)

    st.write("### Nhược điểm")
    st.write("""
    - Khó chọn tham số `eps` và `min_samples` phù hợp.
    - Không hiệu quả với dữ liệu có mật độ thay đổi đáng kể.
    - Tốn thời gian với dữ liệu lớn.
    """)

    st.write("### Ứng dụng")
    st.write("""
    - Phát hiện gian lận trong giao dịch tài chính.
    - Phân tích dữ liệu địa lý.
    - Phân cụm dữ liệu sinh học.
    """)

    st.write("---")

    st.write("## So sánh K-means và DBSCAN")
    st.write("""
    | Đặc điểm               | K-means                           | DBSCAN                            |
    |------------------------|-----------------------------------|-----------------------------------|
    | **Số cụm**             | Cần chỉ định số cụm K trước       | Không cần chỉ định số cụm         |
    | **Hình dạng cụm**      | Chỉ phát hiện cụm hình cầu        | Phát hiện cụm có hình dạng bất kỳ |
    | **Xử lý nhiễu**        | Không xử lý được nhiễu            | Loại bỏ được nhiễu                |
    | **Tham số**            | Chỉ cần số cụm K                  | Cần chọn `eps` và `min_samples`   |
    | **Hiệu suất**          | Nhanh với dữ liệu lớn             | Chậm hơn với dữ liệu lớn          |
    | **Ứng dụng**           | Dữ liệu có hình dạng đơn giản     | Dữ liệu có hình dạng phức tạp     |
    """)

    st.write("## Tài nguyên tham khảo")
    st.write("""
    - [Tài liệu scikit-learn về K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
    - [Tài liệu scikit-learn về DBSCAN](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
    """)

# Streamlit app
def main():
    display_clustering_info()

if __name__ == "__main__":
    main()