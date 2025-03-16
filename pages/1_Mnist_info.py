import streamlit as st

def display_mnist_info():
    st.header("MNIST Dataset")
    st.write("""
      **MNIST** là một trong những bộ dữ liệu nổi tiếng và phổ biến nhất trong cộng đồng học máy, 
      đặc biệt là trong các nghiên cứu về nhận diện mẫu và phân loại hình ảnh.
  
      - Bộ dữ liệu bao gồm tổng cộng **70.000 ảnh chữ số viết tay** từ **0** đến **9**, 
        mỗi ảnh có kích thước **28 x 28 pixel**.
      - Chia thành:
        - **Training set**: 60.000 ảnh để huấn luyện.
        - **Test set**: 10.000 ảnh để kiểm tra.
      - Mỗi hình ảnh là một chữ số viết tay, được chuẩn hóa và chuyển thành dạng grayscale (đen trắng).
  
      Dữ liệu này được sử dụng rộng rãi để xây dựng các mô hình nhận diện chữ số.
      """)

    st.subheader("Một số hình ảnh từ MNIST Dataset")
    st.image("img3.png", caption="Một số hình ảnh từ MNIST Dataset", use_container_width ="auto")

    st.subheader("Ứng dụng thực tế của MNIST")
    st.write("""
      Bộ dữ liệu MNIST đã được sử dụng trong nhiều ứng dụng nhận dạng chữ số viết tay, chẳng hạn như:
      - Nhận diện số trên các hoá đơn thanh toán, biên lai cửa hàng.
      - Xử lý chữ số trên các bưu kiện gửi qua bưu điện.
      - Ứng dụng trong các hệ thống nhận diện tài liệu tự động.
    """)

    st.subheader("Ví dụ về các mô hình học máy với MNIST")
    st.write("""
      Các mô hình học máy phổ biến đã được huấn luyện với bộ dữ liệu MNIST bao gồm:
      - **Logistic Regression**
      - **Decision Trees**
      - **K-Nearest Neighbors (KNN)**
      - **Support Vector Machines (SVM)**
      - **Convolutional Neural Networks (CNNs)**
    """)

    st.subheader("📊 Minh họa dữ liệu MNIST")

    # Mô tả về dữ liệu MNIST
    st.write("""
    Dữ liệu MNIST bao gồm các hình ảnh chữ số viết tay có kích thước **28x28 pixels**.  
    Mỗi ảnh là một **ma trận 28x28**, với mỗi pixel có giá trị từ **0 đến 255**.  
    Khi đưa vào mô hình, ảnh sẽ được biến đổi thành **784 features (28x28)** để làm đầu vào cho mạng nơ-ron.  
    Mô hình sử dụng các lớp ẩn để học và dự đoán chính xác chữ số từ hình ảnh.
    """)

# Streamlit app
def main():
    display_mnist_info()

if __name__ == "__main__":
    main()