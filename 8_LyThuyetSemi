import streamlit as st

# Tiêu đề trang
st.title("Lý Thuyết về Gán Nhãn Giả và Các Bộ Tối Ưu")

# Phần Gán Nhãn Giả
st.header("1. Gán Nhãn Giả (Pseudo Labeling)")
st.write("""
Gán nhãn giả là một kỹ thuật học bán giám sát (semi-supervised learning), trong đó mô hình được huấn luyện trên cả dữ liệu có nhãn và không có nhãn. Cụ thể, mô hình ban đầu được huấn luyện trên dữ liệu có nhãn, sau đó được sử dụng để dự đoán nhãn cho dữ liệu không có nhãn. Các nhãn dự đoán này được gọi là nhãn giả (pseudo labels). Sau đó, mô hình được huấn luyện lại trên cả dữ liệu có nhãn và dữ liệu có nhãn giả.

**Các bước thực hiện:**
1. Huấn luyện mô hình ban đầu trên dữ liệu có nhãn.
2. Sử dụng mô hình để dự đoán nhãn cho dữ liệu không có nhãn.
3. Thêm các nhãn giả vào tập dữ liệu huấn luyện.
4. Huấn luyện lại mô hình trên tập dữ liệu mở rộng.

**Ưu điểm:**
- Tận dụng được dữ liệu không có nhãn, giúp cải thiện hiệu suất mô hình.
- Đặc biệt hữu ích khi dữ liệu có nhãn khan hiếm.

**Nhược điểm:**
- Nhãn giả có thể không chính xác, dẫn đến huấn luyện sai mô hình.
- Cần cân nhắc kỹ lưỡng khi sử dụng nhãn giả để tránh overfitting.
""")

# Phần Bộ Tối Ưu
st.header("2. Các Bộ Tối Ưu Phổ Biến")

# Bộ tối ưu SGD
st.subheader("2.1. SGD (Stochastic Gradient Descent)")
st.write("""
**SGD** là một trong những bộ tối ưu đơn giản và phổ biến nhất. Nó cập nhật các tham số mô hình dựa trên gradient của hàm mất mát đối với từng điểm dữ liệu hoặc từng batch nhỏ.

**Công thức cập nhật:**
\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t; x_i, y_i)
\]
Trong đó:
- \(\theta_t\): Tham số tại bước \(t\).
- \(\eta\): Tốc độ học (learning rate).
- \(\nabla_\theta J(\theta_t; x_i, y_i)\): Gradient của hàm mất mát \(J\) đối với tham số \(\theta\).

**Ưu điểm:**
- Đơn giản và dễ triển khai.
- Hiệu quả với dữ liệu lớn.

**Nhược điểm:**
- Dễ bị mắc kẹt ở các điểm cực tiểu địa phương.
- Dao động nhiều khi tiến đến cực tiểu.
""")

# Bộ tối ưu Adam
st.subheader("2.2. Adam (Adaptive Moment Estimation)")
st.write("""
**Adam** là một bộ tối ưu kết hợp giữa ý tưởng của Momentum và RMSprop. Nó tính toán các giá trị trung bình động của gradient và bình phương gradient để điều chỉnh tốc độ học một cách thích ứng.

**Công thức cập nhật:**
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta J(\theta_t)
\]
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta J(\theta_t))^2
\]
\[
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}
\]
Trong đó:
- \(m_t\): Gradient trung bình động.
- \(v_t\): Bình phương gradient trung bình động.
- \(\beta_1, \beta_2\): Các hệ số động lượng.
- \(\epsilon\): Hằng số nhỏ để tránh chia cho 0.

**Ưu điểm:**
- Hiệu quả cao và thường được sử dụng rộng rãi.
- Tự động điều chỉnh tốc độ học.

**Nhược điểm:**
- Có thể không hiệu quả với một số bài toán cụ thể.
""")

# Bộ tối ưu RMSprop
st.subheader("2.3. RMSprop (Root Mean Square Propagation)")
st.write("""
**RMSprop** là một bộ tối ưu được thiết kế để giải quyết vấn đề tốc độ học giảm dần trong SGD. Nó chia gradient cho căn bậc hai của trung bình động của bình phương gradient.

**Công thức cập nhật:**
\[
v_t = \beta v_{t-1} + (1 - \beta) (\nabla_\theta J(\theta_t))^2
\]
\[
\theta_{t+1} = \theta_t - \eta \frac{\nabla_\theta J(\theta_t)}{\sqrt{v_t} + \epsilon}
\]
Trong đó:
- \(v_t\): Trung bình động của bình phương gradient.
- \(\beta\): Hệ số động lượng.
- \(\epsilon\): Hằng số nhỏ để tránh chia cho 0.

**Ưu điểm:**
- Hiệu quả trong việc điều chỉnh tốc độ học.
- Giảm thiểu dao động khi tiến đến cực tiểu.

**Nhược điểm:**
- Cần điều chỉnh tham số \(\beta\) để đạt hiệu quả tốt nhất.
""")

# Kết thúc trang
st.write("""
Trên đây là một số lý thuyết cơ bản về Gán Nhãn Giả và các bộ tối ưu phổ biến như SGD, Adam, và RMSprop. Hy vọng những thông tin này sẽ hữu ích cho bạn trong quá trình học tập và nghiên cứu!
""")