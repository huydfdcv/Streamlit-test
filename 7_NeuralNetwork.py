import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import time

# ==================== CẤU HÌNH MLFLOW ====================
DAGSHUB_MLFLOW_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2CaXhRNYabm9fN3"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
mlflow.set_experiment("MNIST NeuralNetwork")

# ==================== HÀM VÀ TIỆN ÍCH ====================
@st.cache_resource
def create_model(layer_sizes):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28)))
    for size in layer_sizes:
        model.add(layers.Dense(size, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model

# ==================== GIAO DIỆN CHÍNH ====================
def main():
    st.title("🎨 MNIST Neural Network Trainer")
    
    # Tải dữ liệu
    data = np.load("mnist_data.npz")
    X, y = data['X'], data['y']
    X = X.reshape(-1, 28, 28) if X.shape[1] == 784 else X

    # ==================== TABS CHÍNH ====================
    tab1, tab2, tab3 = st.tabs(["🏋️ Huấn luyện", "📜 Lịch sử", "🎮 Demo"])

    with tab1:
        st.header("Cấu hình Huấn luyện")
        
        with st.form("training_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Tham số Model")
                layer_sizes = st.text_input("Kích thước các layer (ví dụ: 128,64)", "128,64")
                run_name = st.text_input("Tên Run", "Base Model")
                
            with col2:
                st.subheader("Tham số Dữ liệu")
                train_size = st.slider("Số lượng mẫu train", 1000, len(X), 10000)
                epochs = st.slider("Số epoch", 1, 50, 10)
            
            if st.form_submit_button("🚀 Bắt đầu Huấn luyện"):
                with st.spinner("Đang khởi tạo..."):
                    layer_sizes = list(map(int, layer_sizes.split(',')))
                    indices = np.random.choice(len(X), train_size, replace=False)
                    X_train, X_test, y_train, y_test = train_test_split(X[indices], y[indices], test_size=0.2)

                    with mlflow.start_run(run_name=run_name):
                        model = create_model(layer_sizes)
                        model.compile(optimizer='adam',
                                    loss='sparse_categorical_crossentropy',
                                    metrics=['accuracy'])

                        mlflow.log_params({
                            "layer_sizes": layer_sizes,
                            "train_size": train_size,
                            "total_epochs": epochs
                        })

                        history = model.fit(X_train, y_train,
                                          epochs=epochs,
                                          validation_split=0.2)

                        y_pred = np.argmax(model.predict(X_test), axis=1)
                        metrics_dict = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred, average='macro'),
                            "recall": recall_score(y_test, y_pred, average='macro'),
                            "f1": f1_score(y_test, y_pred, average='macro')
                        }
                        mlflow.log_metrics(metrics_dict)
                        mlflow.keras.log_model(model, "model")

                        st.success("✅ Huấn luyện hoàn thành!")

    with tab2:
        st.header("Lịch sử Huấn luyện")
        try:
            runs = mlflow.search_runs()
            st.dataframe(
                runs[['tags.mlflow.runName', 'start_time', 'params.layer_sizes', 
                      'params.total_epochs', 'metrics.accuracy']].rename(
                    columns={
                        'tags.mlflow.runName': 'Tên Run',
                        'start_time': 'Thời gian',
                        'params.layer_sizes': 'Cấu trúc Model',
                        'params.total_epochs': 'Epochs',
                        'metrics.accuracy': 'Độ chính xác'
                    }),
                height=500,
                use_container_width=True  # Sửa thành use_container_width
            )
        except Exception as e:
            st.error(f"⚠️ Lỗi khi tải dữ liệu: {e}")

    with tab3:
        st.header("Demo Nhận dạng Chữ số")
        st.write("Nếu trang vẽ không xuất hiện lần đầu tiên thì nếu reload lại trang thì nó có thể xuất hiện lạilại")
        try:
            # Phần chọn run đầu tiên
            runs = mlflow.search_runs()
            run_names = runs['tags.mlflow.runName'].tolist()
            selected_run = st.selectbox("Chọn Model để Demo", run_names)
            
            if selected_run:
                # Tạo canvas
                canvas_result = st_canvas(
                    fill_color="rgba(0, 0, 0, 1)",
                    stroke_width=20,
                    stroke_color="rgba(255, 255, 255, 1)",
                    background_color="#000000",
                    height=280,
                    width=280,
                    drawing_mode="freedraw",
                    key="canvas"
                )
                
                # Nút dự đoán
                if st.button("🔮 Dự đoán", type="primary"):
                    # Xử lý ảnh
                    if canvas_result.image_data is not None:
                        img = Image.fromarray(canvas_result.image_data.astype('uint8'))
                        img = img.resize((28, 28)).convert('L')
                        img_array = np.array(img) / 255.0
                        img_array = 1 - img_array  # Đảo ngược màu
                        
                        # Load model và dự đoán
                        run_id = runs[runs['tags.mlflow.runName'] == selected_run]['run_id'].iloc[0]
                        model = mlflow.keras.load_model(f"runs:/{run_id}/model")
                        prediction = model.predict(img_array.reshape(1, 28, 28))
                        
                        # Hiển thị kết quả
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(img_array, width=150, caption="Ảnh đã xử lý")
                        with col2:
                            st.subheader(f"Dự đoán: {np.argmax(prediction)}")
                            st.metric("Độ tin cậy", f"{np.max(prediction)*100:.2f}%")
                            st.bar_chart(prediction[0], height=200)
        except Exception as e:
            st.warning("Chưa có model nào được huấn luyện hoặc có lỗi xảy ra!")

if __name__ == "__main__":
    main()
    