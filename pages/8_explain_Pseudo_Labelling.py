import streamlit as st
def show_prediction_table():
    st.table({
        "Ảnh": ["Ảnh 1", "Ảnh 2", "Ảnh 3", "Ảnh 4", "Ảnh 5"],
        "Dự đoán": [7, 2, 3, 5, 8],
        "Xác suất": [0.98, 0.85, 0.96, 0.88, 0.97],
        "Gán nhãn?": ["✅", "❌", "✅", "❌", "✅"]
    })

def explain_Pseudo_Labelling():
    
    
    st.markdown("## 📚 Lý thuyết về Pseudo Labelling")
    st.write("""
    **Pseudo Labelling** là một phương pháp semi-supervised learning giúp kết hợp dữ liệu có nhãn và không nhãn để cải thiện độ chính xác của mô hình. Quá trình này diễn ra qua các bước sau:
    
    1️⃣ **Huấn luyện mô hình ban đầu** trên một tập dữ liệu nhỏ (~1% tổng số dữ liệu có nhãn).  
    2️⃣ **Dự đoán nhãn** cho các mẫu chưa được gán nhãn bằng mô hình đã huấn luyện.  
    3️⃣ **Lọc các dự đoán có độ tin cậy cao** dựa trên ngưỡng xác suất (ví dụ: > 0.95).  
    4️⃣ **Gán nhãn giả (Pseudo Labels)** cho các mẫu tin cậy.  
    5️⃣ **Thêm dữ liệu đã gán nhãn giả vào tập train**, mở rộng dữ liệu huấn luyện.  
    6️⃣ **Huấn luyện lại mô hình** với tập dữ liệu mở rộng (gồm dữ liệu ban đầu + dữ liệu có nhãn giả).  
    7️⃣ **Lặp lại các bước trên** cho đến khi đạt điều kiện dừng (hội tụ hoặc số lần lặp tối đa).  
    """)

    # st.image("buoi8/img1.png", caption="Các bước Pseudo Labelling", use_container_width ="auto")

    # Ví dụ minh họa
    st.markdown("## 🔍 Ví dụ về Pseudo Labelling")
    st.write("""
    Giả sử ta có 70.000 ảnh chữ số viết tay (0-9), nhưng chỉ có 1% (100 ảnh) với mỗi số được gán nhãn ban đầu.  
    → Còn lại 60.000 ảnh không nhãn.
    """)

    st.markdown("### 🏁 **Bước 1: Huấn luyện mô hình ban đầu**")
    st.write("""
    - Mô hình được train trên 1000 ảnh có nhãn.  
    - Do dữ liệu ít, mô hình có độ chính xác thấp.  
    """)

    st.markdown("### 🧠 **Bước 2: Dự đoán nhãn cho dữ liệu chưa gán nhãn**")
    st.write("""
    - Chạy mô hình trên 60.000 ảnh chưa gán nhãn.  
    - Dự đoán và tính xác suất cho mỗi ảnh.  
    """)
    
    show_prediction_table()  # Hiển thị bảng dự đoán mẫu

    st.markdown("### 🔬 **Bước 3: Lọc dữ liệu có độ tin cậy cao**")
    st.write("""
    - Chỉ chọn những ảnh có xác suất dự đoán cao hơn ngưỡng tin cậy (ví dụ: 0.95).  
    - Ảnh 1, 3, 5 sẽ được gán nhãn giả.
    - Ảnh 2, 4 bị bỏ qua vì mô hình không tự tin.
    - Những ảnh đạt tiêu chuẩn sẽ được gán nhãn giả.  
    - Ảnh có độ tin cậy thấp sẽ bị loại bỏ.  
    """)

    st.markdown("### 🏷️ **Bước 4: Gán nhãn giả cho các dự đoán tin cậy**")
    st.write("""
    - Các mẫu có độ tin cậy cao được gán nhãn theo kết quả dự đoán của mô hình.  
    - ví dụ có 500 ảnh được gán nhãn giả.
    """)

    st.markdown("### 📂 **Bước 5: Thêm dữ liệu gán nhãn giả vào tập train**")
    st.write("""
    - Tập train mới = dữ liệu ban đầu + các ảnh có nhãn giả.  
    - Ví dụ: từ 1000 ảnh có nhãn ban đầu, ta có thể mở rộng lên 1500 ảnh sau khi thêm nhãn giả.  
    """)

    st.markdown("### 🔄 **Bước 6: Huấn luyện lại mô hình với tập dữ liệu mở rộng**")
    st.write("""
    - Huấn luyện lại mô hình trên tập dữ liệu mới.  
    - Mô hình sẽ học thêm từ dữ liệu mới và dần cải thiện độ chính xác.  
    """)

    st.markdown("### 🔁 **Bước 7: Lặp lại quá trình đến khi hội tụ**")
    st.write("""
    - Quá trình tiếp tục cho đến khi đạt điều kiện dừng:  
      - Đạt số lần lặp tối đa  
      - Mô hình không cải thiện thêm  
    """)

    st.markdown("## 🎯 **Kết quả cuối cùng**")
    st.write("""
    - Ban đầu chỉ có 100 ảnh có nhãn.  
    - Sau vài vòng lặp, mô hình có thể tự gán nhãn cho hàng ngàn ảnh.  
    - Độ chính xác tăng dần theo mỗi lần huấn luyện lại.  
    """)


