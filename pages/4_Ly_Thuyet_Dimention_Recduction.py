import streamlit as st

st.title("Giải thích về PCA và t-SNE")

st.header("1. Principal Component Analysis (PCA)")
st.write(
    "PCA (Phân tích thành phần chính) là một kỹ thuật giảm chiều dữ liệu bằng cách tìm các thành phần chính có phương sai cao nhất trong dữ liệu. Dưới đây là các bước thực hiện PCA:"
)

st.write("**Bước 1: Chuẩn hóa dữ liệu**")
st.latex(r"""
X_{norm} = \frac{X - \mu}{\sigma}
""")
st.write("Trong đó:")
st.write("- \( X \): Dữ liệu ban đầu.")
st.write("- \( \mu \): Giá trị trung bình của dữ liệu.")
st.write("- \( \sigma \): Độ lệch chuẩn của dữ liệu.")

st.write("**Bước 2: Tính ma trận hiệp phương sai**")
st.latex(r"""
\Sigma = \frac{1}{n} X_{norm}^T X_{norm}
""")
st.write("Trong đó:")
st.write("- \( \Sigma \): Ma trận hiệp phương sai.")
st.write("- \( n \): Số lượng mẫu.")

st.write("**Bước 3: Tính toán giá trị riêng và vector riêng**")
st.latex(r"""
\Sigma W = \lambda W
""")
st.write("Trong đó:")
st.write("- \( W \): Ma trận eigenvector.")
st.write("- \( \lambda \): Eigenvalue tương ứng.")

st.write("**Bước 4: Chuyển đổi dữ liệu sang không gian mới**")
st.latex(r"""
X' = X_{norm} W_k
""")
st.write("Trong đó:")
st.write("- \( W_k \): Ma trận chứa \( k \) vector riêng ứng với \( k \) giá trị riêng lớn nhất.")

st.header("2. t-Distributed Stochastic Neighbor Embedding (t-SNE)")
st.write(
    "t-SNE là một thuật toán giảm chiều dữ liệu phi tuyến tính, giúp bảo toàn cấu trúc cục bộ của dữ liệu. Dưới đây là cách t-SNE hoạt động:"
)

st.write("**Bước 1: Xác suất đồng dạng trong không gian gốc**")
st.latex(r"""
P_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2 \sigma_i^2)}{\sum_{k \neq i} \exp(-\|x_i - x_k\|^2 / 2 \sigma_i^2)}
""")
st.latex(r"""
P_{ij} = \frac{P_{j|i} + P_{i|j}}{2n}
""")
st.write("Trong đó:")
st.write("- \( P_{j|i} \): Xác suất đồng dạng giữa điểm \( x_i \) và \( x_j \).")
st.write("- \( \sigma_i \): Tham số độ rộng của phân bố Gaussian tại điểm \( x_i \).")
st.write("- \( n \): Số lượng mẫu.")

st.write("**Bước 2: Xác suất trong không gian nhúng**")
st.latex(r"""
Q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
""")
st.write("Trong đó:")
st.write("- \( Q_{ij} \): Xác suất đồng dạng giữa điểm \( y_i \) và \( y_j \) trong không gian nhúng.")

st.write("**Bước 3: Giảm thiểu hàm mất mát KL-divergence**")
st.latex(r"""
C = \sum_{i} \sum_{j} P_{ij} \log \frac{P_{ij}}{Q_{ij}}
""")
st.write("Trong đó:")
st.write("- \( C \): Hàm mất mát KL-divergence.")
st.write("- Gradient được tính toán để cập nhật tọa độ \( y_i \) nhằm tối ưu hóa sự tương đồng giữa \( P_{ij} \) và \( Q_{ij} \).")