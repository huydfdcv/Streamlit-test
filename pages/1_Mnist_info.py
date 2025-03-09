import streamlit as st

def display_mnist_info():
    st.title("Giới thiệu về tập dữ liệu MNIST")
    
    st.write("""
    ## Tập dữ liệu MNIST là gì?
    MNIST (Modified National Institute of Standards and Technology) là một tập dữ liệu phổ biến trong lĩnh vực học máy và thị giác máy tính.
    - **Mục đích**: Nhận dạng chữ số viết tay từ 0 đến 9.
    - **Kích thước**: 70.000 ảnh (60.000 ảnh huấn luyện và 10.000 ảnh kiểm tra).
    - **Độ phân giải**: Mỗi ảnh có kích thước 28x28 pixel.
    """)


    st.write("## Ví dụ về dữ liệu MNIST")
    st.write("Dưới đây là ảnh từ tập dữ liệu MNIST:")
    st.image("img3.png")

    st.write("""
    ## Ứng dụng của MNIST
    - **Nhận dạng chữ số viết tay**: MNIST thường được sử dụng để huấn luyện và đánh giá các mô hình nhận dạng chữ số.
    - **Benchmark**: MNIST là một tập dữ liệu benchmark phổ biến để so sánh hiệu suất của các thuật toán học máy.
    - **Giáo dục**: MNIST thường được sử dụng trong các khóa học về học máy và thị giác máy tính do tính đơn giản và dễ hiểu.
    """)

    st.write("## Cách sử dụng MNIST trong Python")
    st.write("""
    Bạn có thể tải tập dữ liệu MNIST bằng thư viện `scikit-learn`:
    ```python
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    X, y = mnist.data, mnist.target
    ```
    - `X`: Dữ liệu ảnh (mỗi ảnh là một vector 784 chiều).
    - `y`: Nhãn (chữ số từ 0 đến 9).
    """)

    st.write("## Tài nguyên tham khảo")
    st.write("""
    - [Trang chủ MNIST](http://yann.lecun.com/exdb/mnist/)
    - [Tài liệu scikit-learn về MNIST](https://scikit-learn.org/stable/datasets/toy_dataset.html#mnist-dataset)
    """)

# Streamlit app
def main():
    display_mnist_info()

if __name__ == "__main__":
    main()
