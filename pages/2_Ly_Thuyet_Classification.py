import streamlit as st

# Hiển thị thông tin về Decision Tree và SVM
def display_algorithms_info():
    st.title("Giới thiệu về Decision Tree và SVM với công thức toán học")

    st.write("""
    ## Decision Tree (Cây quyết định)
    **Cây quyết định** là một thuật toán học máy được sử dụng cho cả bài toán phân loại (classification) và hồi quy (regression).
    """)

    st.write("### Cách hoạt động")
    st.write("""
    - Cây quyết định chia dữ liệu thành các nhóm nhỏ dựa trên các đặc trưng (features).
    - Mỗi nút trong cây đại diện cho một đặc trưng, và mỗi nhánh đại diện cho một quyết định dựa trên giá trị của đặc trưng đó.
    - Quá trình chia dữ liệu tiếp tục cho đến khi đạt được các nhóm đồng nhất (homogeneous) hoặc đạt điều kiện dừng.
    """)

    st.write("### Công thức toán học")
    st.write("""
    #### 1. **Entropy** (Độ hỗn loạn)
    Entropy đo lường độ hỗn loạn của dữ liệu tại một nút:
    $$
    Entropy(S) = -\\sum_{i=1}^{c} p_i \\log_2(p_i)
    $$
    Trong đó:
    - \( S \): Tập dữ liệu tại nút.
    - \( c \): Số lớp.
    - \( p_i \): Tỷ lệ của lớp \( i \) trong tập dữ liệu.

    #### 2. **Information Gain** (Độ lợi thông tin)
    Information Gain đo lường sự giảm entropy sau khi chia dữ liệu dựa trên một đặc trưng:
    $$
    IG(S, A) = Entropy(S) - \\sum_{v \\in Values(A)} \\frac{|S_v|}{|S|} Entropy(S_v)
    $$
    Trong đó:
    - \( A \): Đặc trưng được sử dụng để chia dữ liệu.
    - \( Values(A) \): Các giá trị có thể của đặc trưng \( A \).
    - \( S_v \): Tập con của \( S \) với giá trị \( v \) của đặc trưng \( A \).
    """)

    st.write("### Ưu điểm")
    st.write("""
    - Dễ hiểu và dễ diễn giải.
    - Không cần chuẩn hóa dữ liệu.
    - Xử lý được cả dữ liệu số và dữ liệu phân loại.
    """)

    st.write("### Nhược điểm")
    st.write("""
    - Dễ bị overfitting nếu cây quá phức tạp.
    - Nhạy cảm với sự thay đổi nhỏ trong dữ liệu.
    - Có thể tạo ra cây không cân bằng nếu dữ liệu không đồng đều.
    """)

    st.write("### Ứng dụng")
    st.write("""
    - Phân loại văn bản.
    - Dự đoán rủi ro tín dụng.
    - Chẩn đoán y tế.
    """)

    st.write("---")

    st.write("""
    ## Support Vector Machine (SVM)
    **SVM** là một thuật toán học máy mạnh mẽ được sử dụng cho cả bài toán phân loại và hồi quy.
    """)

    st.write("### Cách hoạt động")
    st.write("""
    - SVM tìm một siêu phẳng (hyperplane) tối ưu để phân tách các lớp dữ liệu.
    - Siêu phẳng được chọn sao cho khoảng cách (margin) giữa các lớp là lớn nhất.
    - SVM có thể sử dụng kernel để xử lý dữ liệu không phân tách tuyến tính.
    """)

    st.write("### Công thức toán học")
    st.write("""
    #### 1. **Siêu phẳng phân tách**
    Siêu phẳng phân tách được định nghĩa bởi phương trình:
    $$
    w^T x + b = 0
    $$
    Trong đó:
    - \( w \): Vector trọng số.
    - \( x \): Vector đặc trưng.
    - \( b \): Hệ số bias.

    #### 2. **Bài toán tối ưu**
    SVM tối ưu hóa bài toán sau để tìm siêu phẳng tối ưu:
    $$
    \\min_{w, b} \\frac{1}{2} \\|w\\|^2
    $$
    Với ràng buộc:
    $$
    y_i (w^T x_i + b) \\geq 1, \\quad \\forall i
    $$
    Trong đó:
    - \( y_i \): Nhãn của điểm dữ liệu \( x_i \).

    #### 3. **Kernel Trick**
    SVM sử dụng kernel để ánh xạ dữ liệu vào không gian nhiều chiều:
    $$
    K(x_i, x_j) = \\phi(x_i)^T \\phi(x_j)
    $$
    Trong đó:
    - \( \\phi \): Hàm ánh xạ.
    - Các kernel phổ biến: Linear, Polynomial, RBF.
    """)

    st.write("### Ưu điểm")
    st.write("""
    - Hiệu quả trong không gian nhiều chiều.
    - Linh hoạt nhờ sử dụng kernel.
    - Ít bị ảnh hưởng bởi overfitting nếu chọn kernel phù hợp.
    """)

    st.write("### Nhược điểm")
    st.write("""
    - Khó diễn giải hơn so với Decision Tree.
    - Cần lựa chọn kernel và tham số phù hợp.
    - Tốn thời gian huấn luyện với dữ liệu lớn.
    """)

    st.write("### Ứng dụng")
    st.write("""
    - Nhận dạng chữ viết tay.
    - Phân loại ảnh.
    - Phân tích sinh học (ví dụ: phân loại protein).
    """)

    st.write("---")

    st.write("## So sánh Decision Tree và SVM")
    st.write("""
    | Đặc điểm               | Decision Tree                     | SVM                               |
    |------------------------|-----------------------------------|-----------------------------------|
    | **Diễn giải**           | Dễ diễn giải                     | Khó diễn giải hơn                |
    | **Xử lý dữ liệu**       | Xử lý được cả số và phân loại    | Cần chuẩn hóa dữ liệu            |
    | **Overfitting**         | Dễ bị overfitting                | Ít bị overfitting nếu chọn kernel phù hợp |
    | **Thời gian huấn luyện**| Nhanh với dữ liệu nhỏ            | Chậm hơn với dữ liệu lớn         |
    | **Ứng dụng**            | Phân loại đơn giản, dữ liệu nhỏ | Phân loại phức tạp, dữ liệu lớn  |
    """)

    st.write("## Tài nguyên tham khảo")
    st.write("""
    - [Tài liệu scikit-learn về Decision Tree](https://scikit-learn.org/stable/modules/tree.html)
    - [Tài liệu scikit-learn về SVM](https://scikit-learn.org/stable/modules/svm.html)
    """)

# Streamlit app
def main():
    display_algorithms_info()

if __name__ == "__main__":
    main()