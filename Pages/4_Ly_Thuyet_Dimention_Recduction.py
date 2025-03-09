import streamlit as st

st.title("Giải thích về PCA và t-SNE")

st.header("1. Principal Component Analysis (PCA)")
st.write(
    "PCA (Phân tích thành phần chính) là một kỹ thuật giảm chiều dữ liệu bằng cách tìm các thành phần chính có phương sai cao nhất trong dữ liệu. Dưới đây là các bước thực hiện PCA:\n\n"
    "**Bước 1: Chuẩn hóa dữ liệu**\n"
    "Dữ liệu ban đầu \( X \) được chuẩn hóa bằng cách trừ trung bình và chia cho độ lệch chuẩn:\n\n"
    "\[ X_{norm} = \frac{X - \mu}{\sigma} \]\n\n"
    "**Bước 2: Tính ma trận hiệp phương sai**\n"
    "\[ \Sigma = \frac{1}{n} X_{norm}^T X_{norm} \]\n\n"
    "**Bước 3: Tính toán giá trị riêng và vector riêng**\n"
    "Giải bài toán:\n\n"
    "\[ \Sigma W = \lambda W \]\n\n"
    "Với \( W \) là ma trận eigenvector và \( \lambda \) là eigenvalue tương ứng.\n\n"
    "**Bước 4: Chuyển đổi dữ liệu sang không gian mới**\n"
    "\[ X' = X_{norm} W_k \]\n\n"
    "Với \( W_k \) chứa \( k \) vector riêng ứng với \( k \) giá trị riêng lớn nhất."
)

st.header("2. t-Distributed Stochastic Neighbor Embedding (t-SNE)")
st.write(
    "t-SNE là một thuật toán giảm chiều dữ liệu phi tuyến tính, giúp bảo toàn cấu trúc cục bộ của dữ liệu. Dưới đây là cách t-SNE hoạt động:\n\n"
    "**Bước 1: Xác suất đồng dạng trong không gian gốc**\n"
    "\[ P_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)} \]\n\n"
    "Xác suất tổng hợp:\n\n"
    "\[ P_{ij} = \frac{P_{j|i} + P_{i|j}}{2n} \]\n\n"
    "**Bước 2: Xác suất trong không gian nhúng**\n"
    "\[ Q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}} \]\n\n"
    "**Bước 3: Giảm thiểu hàm mất mát KL-divergence**\n"
    "Hàm mất mát của t-SNE:\n\n"
    "\[ C = \sum_{i} \sum_{j} P_{ij} \log \frac{P_{ij}}{Q_{ij}} \]\n\n"
    "Gradient được tính toán để cập nhật tọa độ \( y_i \) nhằm tối ưu hóa sự tương đồng giữa \( P_{ij} \) và \( Q_{ij} \)."
)

