import streamlit as st
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.datasets import fetch_openml

# Thiết lập MLflow với DagsHub
DAGSHUB_MLFLOW_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2CaXhRNYabm9fN3"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
mlflow.set_experiment("MNIST Pseudo Labelling")

# Hàm tải dữ liệu MNIST từ OpenML
def tai_du_lieu():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    x, y = mnist.data, mnist.target.astype(np.uint8)
    return x / 255.0, y  # Chuẩn hóa dữ liệu

# Hàm tạo mô hình mạng neural
def tao_mo_hinh():
    return models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

# Lấy 1% dữ liệu từ mỗi class
def lay_du_lieu_khoi_tao(x_train, y_train, phan_tram=0.01):
    x_khoi_tao, y_khoi_tao = [], []
    for lop in range(10):
        vi_tri = np.where(y_train == lop)[0]
        chon = np.random.choice(vi_tri, int(len(vi_tri) * phan_tram), replace=False)
        x_khoi_tao.append(x_train[chon])
        y_khoi_tao.append(y_train[chon])
    return np.concatenate(x_khoi_tao), np.concatenate(y_khoi_tao)

# Gán nhãn giả
def gan_nhan_gia(mo_hinh, x_chua_nhan, nguong=0.95):
    if len(x_chua_nhan) == 0:
        return np.array([]), np.array([])  # Trả về mảng rỗng nếu không có dữ liệu
    xac_suat = mo_hinh.predict(x_chua_nhan)
    nhan_du_doan = np.argmax(xac_suat, axis=1)
    xac_suat_cao_nhat = np.max(xac_suat, axis=1)
    mask = xac_suat_cao_nhat >= nguong
    return x_chua_nhan[mask], nhan_du_doan[mask]

# Quá trình huấn luyện với gán nhãn giả
def huan_luyen_voi_nhan_gia(x_train, y_train, x_test, y_test, nguong=0.95, so_vong_lap_toi_da=10, so_epoch=5, dieu_kien_dung="so_vong_lap"):
    # Lấy 1% dữ liệu ban đầu
    x_hien_tai, y_hien_tai = lay_du_lieu_khoi_tao(x_train, y_train)
    x_chua_nhan = np.delete(x_train, np.where(np.isin(x_train, x_hien_tai).all(axis=1)), axis=0)
    
    mo_hinh = tao_mo_hinh()
    toc_do_hoc = st.session_state.toc_do_hoc
    bo_toi_uu = st.session_state.bo_toi_uu
    
    # Thiết lập bộ tối ưu
    if bo_toi_uu == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=toc_do_hoc)
    elif bo_toi_uu == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=toc_do_hoc)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=toc_do_hoc)
    
    mo_hinh.compile(optimizer=optimizer, 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Theo dõi tiến trình
    thanh_tien_trinh = st.progress(0)
    trang_thai = st.empty()
    
    vong = 0
    while True:
        vong += 1
        
        # Kiểm tra điều kiện dừng
        if dieu_kien_dung == "so_vong_lap" and vong > so_vong_lap_toi_da:
            st.write(f"Đã đạt số vòng lặp tối đa: {so_vong_lap_toi_da}")
            break
        if dieu_kien_dung == "gan_het_nhan" and len(x_chua_nhan) == 0:
            st.write("Đã gán hết nhãn cho tập dữ liệu.")
            break
        
        # Log thông số cho từng vòng lặp
        mlflow.log_param("iteration", vong)
        
        # Huấn luyện từng epoch
        for epoch in range(so_epoch):
            ti_le = ((vong - 1) * so_epoch + epoch + 1) / (so_vong_lap_toi_da * so_epoch)
            thanh_tien_trinh.progress(ti_le)
            trang_thai.text(f"Vòng {vong}/{so_vong_lap_toi_da} - Epoch {epoch + 1}/{so_epoch}")
            mo_hinh.fit(x_hien_tai, y_hien_tai, epochs=1, batch_size=32, 
                      validation_data=(x_test, y_test), verbose=0)
        
        # Gán nhãn giả
        x_nhan_gia, y_nhan_gia = gan_nhan_gia(mo_hinh, x_chua_nhan, nguong)
        if len(x_nhan_gia) == 0:
            st.write(f"Vòng {vong}: Không còn mẫu nào đạt ngưỡng")
            break
            
        # Cập nhật dữ liệu huấn luyện
        x_hien_tai = np.concatenate([x_hien_tai, x_nhan_gia])
        y_hien_tai = np.concatenate([y_hien_tai, y_nhan_gia])
        x_chua_nhan = np.delete(x_chua_nhan, np.where(np.isin(x_chua_nhan, x_nhan_gia).all(axis=1)), axis=0)
        
        # Log metrics
        mlflow.log_metric("train_size", len(x_hien_tai))
        mlflow.log_metric("pseudo_labels_added", len(x_nhan_gia))
        
        # Đánh giá mô hình
        do_mat, do_chinh_xac = mo_hinh.evaluate(x_test, y_test, verbose=0)
        mlflow.log_metric("test_accuracy", do_chinh_xac)
        st.write(f"Vòng {vong}: Độ chính xác = {do_chinh_xac:.4f}")
    
    return mo_hinh

# Giao diện chính
def main():
    st.title("Ứng dụng Gán Nhãn Giả cho MNIST")
    
    # Tải dữ liệu MNIST
    x, y = tai_du_lieu()
    
    # Chia tập train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Hiển thị mẫu dữ liệu
    st.subheader("Mẫu dữ liệu huấn luyện")
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Nhãn: {y_train[i]}")
        axes[i].axis('off')
    st.pyplot(fig)
    
    # Thiết lập các tham số
    st.subheader("Thông số huấn luyện")
    nguong = st.slider("Ngưỡng tin cậy", 0.8, 1.0, 0.95,
                      help="Ngưỡng xác suất tối thiểu để chấp nhận nhãn giả")
    so_vong_lap_toi_da = st.slider("Số vòng lặp tối đa", 1, 20, 10)
    so_epoch = st.slider("Số epoch mỗi vòng", 1, 10, 5)
    toc_do_hoc = st.number_input("Tốc độ học", value=0.001, format="%.4f",
                                help="Tốc độ học cho thuật toán tối ưu")
    bo_toi_uu = st.selectbox("Bộ tối ưu", ["Adam", "SGD", "RMSprop"],
                            help="Lựa chọn thuật toán tối ưu hóa")
    dieu_kien_dung = st.radio("Điều kiện dừng", 
                             ["Dừng sau số vòng lặp tối đa", "Dừng khi gán hết nhãn"],
                             help="Chọn điều kiện để dừng quá trình huấn luyện")
    ten_run = st.text_input("Tên run", "Run mặc định")
    
    # Lưu tham số vào session state
    st.session_state.toc_do_hoc = toc_do_hoc
    st.session_state.bo_toi_uu = bo_toi_uu
    
    # Bắt đầu huấn luyện
    if st.button("Bắt đầu huấn luyện"):
        with mlflow.start_run(run_name=ten_run):
            # Log các tham số chính
            mlflow.log_params({
                "threshold": nguong,
                "max_iterations": so_vong_lap_toi_da,
                "epochs_per_iteration": so_epoch,
                "learning_rate": toc_do_hoc,
                "optimizer": bo_toi_uu,
                "stop_condition": dieu_kien_dung
            })
            
            mo_hinh = huan_luyen_voi_nhan_gia(
                x_train, y_train, x_test, y_test,
                nguong, so_vong_lap_toi_da, so_epoch,
                dieu_kien_dung="so_vong_lap" if dieu_kien_dung == "Dừng sau số vòng lặp tối đa" else "gan_het_nhan"
            )
            
            # Đánh giá cuối cùng
            do_mat, do_chinh_xac = mo_hinh.evaluate(x_test, y_test, verbose=0)
            st.subheader("Kết quả cuối cùng")
            st.write(f"Độ chính xác: {do_chinh_xac:.4f}")
            st.write(f"Giá trị mất mát: {do_mat:.4f}")
            mlflow.log_metrics({
                "final_test_accuracy": do_chinh_xac,
                "final_test_loss": do_mat
            })
            st.session_state.mo_hinh = mo_hinh  # Lưu mô hình để demo

if __name__ == "__main__":
    main()