import streamlit as st
import numpy as np
import mlflow
import os
import time
from datetime import datetime
import time
import mlflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from mlflow.models.signature import infer_signature
import random
import pandas as pd
import time
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
# Load dữ liệu MNIST
def load_mnist_data():
    X = np.load("X.npy")
    y = np.load("y.npy")
    return X, y

def split_data():
    st.title("📌 Chia dữ liệu Train/Test")
    
    # Đọc dữ liệu
    X, y = load_mnist_data()
    total_samples = X.shape[0] 
    
    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("📌 Chọn số lượng ảnh để huấn luyện:", 1000, total_samples, 10000)
    num_samples =num_samples -10
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("📌 Chọn % dữ liệu Test", 10, 50, 20)
    train_size = 100 - test_size
    
    st.write(f"📌 **Tỷ lệ phân chia:** Test={test_size}%, Train={train_size}%")
    
    if st.button("✅ Xác nhận & Lưu"):
        X_selected, _, y_selected, _ = train_test_split(X, y, train_size=num_samples, stratify=y, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42)
        
        # Lưu vào session_state
        st.session_state.update({
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test
        })
        
        summary_df = pd.DataFrame({"Tập dữ liệu": ["Train", "Test"], "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]]})
        st.success("✅ Dữ liệu đã được chia thành công!")
        st.table(summary_df)

def thi_nghiem():
    num = 0
    if "X_train" not in st.session_state:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    X_train, X_test = [st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_test"]]
    y_train, y_test = [st.session_state[k] for k in ["y_train", "y_test"]]
    st.title(f"Chọn tham số cho Neural Network ")
    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    num_layers = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    optimizer = st.selectbox("Optimizer:", ["adam", "sgd", "rmsprop"])
    epochs = st.slider("🕰 Số epochs:", min_value=1, max_value=50, value=20, step=1)
    learning_rate = st.slider("⚡ Tốc độ học (Learning Rate):", min_value=1e-5, max_value=1e-1, value=1e-3, step=1e-5, format="%.5f")
    
    st.title(f"Chọn tham số cho Pseudo Labelling ")
    labeled_ratio = st.slider("📊 Tỉ lệ dữ liệu có nhãn ban đầu (%):", min_value=1, max_value=20, value=1, step=1)
    max_iterations = st.slider("🔄 Số lần lặp tối đa của Pseudo-Labeling:", min_value=1, max_value=10, value=3, step=1)
    confidence_threshold = st.slider("✅ Ngưỡng tin cậy Pseudo Labeling (%):", min_value=50, max_value=99, value=95, step=1) / 100.0

    loss_fn = "sparse_categorical_crossentropy"
    run_name = st.text_input("🔹 Nhập tên Run:", "Default_Run")
    st.session_state['run_name'] = run_name

    if st.button("🚀 Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)
            mlflow.log_params({
                "num_layers": num_layers,
                "num_neurons": num_neurons,
                "activation": activation,
                "optimizer": optimizer,
                "learning_rate": learning_rate,
                "k_folds": k_folds,
                "epochs": epochs,
                "labeled_ratio": labeled_ratio,
                "max_iterations": max_iterations,
                "confidence_threshold": confidence_threshold
            })

            num_labeled = int(len(X_train) * labeled_ratio / 100)
            labeled_idx = np.random.choice(len(X_train), num_labeled, replace=False)
            unlabeled_idx = np.setdiff1d(np.arange(len(X_train)), labeled_idx)

            X_labeled, y_labeled = X_train[labeled_idx], y_train[labeled_idx]
            X_unlabeled = X_train[unlabeled_idx]

            total_pseudo_labels = 0  # Tổng số nhãn giả được thêm vào

            for iteration in range(max_iterations):
                kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
                accuracies, losses = [], []
                training_progress = st.progress(0)
                training_status = st.empty()

                num = 0
                total_steps = k_folds * max_iterations

                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_labeled, y_labeled)):
                    X_k_train, X_k_val = X_labeled[train_idx], X_labeled[val_idx]
                    y_k_train, y_k_val = y_labeled[train_idx], y_labeled[val_idx]

                    model = keras.Sequential([
                        layers.Input(shape=(X_k_train.shape[1],))
                    ] + [
                        layers.Dense(num_neurons, activation=activation) for _ in range(num_layers)
                    ] + [
                        layers.Dense(10, activation="softmax")
                    ])

                    if optimizer == "adam":
                        opt = keras.optimizers.Adam(learning_rate=learning_rate)
                    elif optimizer == "sgd":
                        opt = keras.optimizers.SGD(learning_rate=learning_rate)
                    else:
                        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

                    model.compile(optimizer=opt, loss=loss_fn, metrics=["accuracy"])

                    start_time = time.time()
                    history = model.fit(X_k_train, y_k_train, epochs=epochs, validation_data=(X_k_val, y_k_val), verbose=0)
                    elapsed_time = time.time() - start_time

                    accuracies.append(history.history["val_accuracy"][-1])
                    losses.append(history.history["val_loss"][-1])
                    num += 1
                    progress_percent = int((num / k_folds) * 100)

                    training_progress.progress(progress_percent)
                    training_status.text(f"⏳ Đang huấn luyện... {progress_percent}%")

                avg_val_accuracy = np.mean(accuracies)
                avg_val_loss = np.mean(losses)

                mlflow.log_metrics({
                    "avg_val_accuracy": avg_val_accuracy,
                    "avg_val_loss": avg_val_loss,
                    "elapsed_time": elapsed_time
                })

                pseudo_preds = model.predict(X_unlabeled)
                pseudo_labels = np.argmax(pseudo_preds, axis=1)
                confidence_scores = np.max(pseudo_preds, axis=1)
                confident_mask = confidence_scores > confidence_threshold

                num_pseudo_added = np.sum(confident_mask)
                total_pseudo_labels += num_pseudo_added

                X_labeled = np.concatenate([X_labeled, X_unlabeled[confident_mask]])
                y_labeled = np.concatenate([y_labeled, pseudo_labels[confident_mask]])
                X_unlabeled = X_unlabeled[~confident_mask]

                # Đánh giá mô hình trên tập validation và test sau khi gán nhãn giả
                #val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

                st.write(f"📢 **Vòng lặp {iteration+1}:**")
                st.write(f"- Số pseudo labels mới thêm: {num_pseudo_added}")
                st.write(f"- Tổng số pseudo labels: {total_pseudo_labels}")
                st.write(f"- Số lượng dữ liệu chưa gán nhãn còn lại: {len(X_unlabeled)}")
                # st.write(f"- 🔥 **Độ chính xác trên tập validation:** {val_accuracy:.4f}")
                st.write(f"- 🚀 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
                st.write("---")

                # Lưu độ chính xác vào MLflow để theo dõi
                mlflow.log_metrics({
                    # f"val_accuracy_iter_{iteration+1}": val_accuracy,
                    f"test_accuracy_iter_{iteration+1}": test_accuracy
                })
                if len(X_unlabeled) == 0:
                    break

            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            mlflow.log_metrics({"test_accuracy": test_accuracy, "test_loss": test_loss})
            mlflow.end_run()
            st.session_state["trained_model"] = model
            training_progress.progress(100)
            training_status.text("✅ Huấn luyện hoàn tất!")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(f"📊 **Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
            st.write(f"📊 **Độ chính xác trên tập test:** {test_accuracy:.4f}")
            st.success(f"✅ Đã log dữ liệu cho **{st.session_state['run_name']}** trong MLflow! 🚀")
            st.markdown(f"🔗 [Truy cập MLflow UI]({st.session_state['mlflow_url']})")
def preprocess_canvas_image(canvas_result):
    """Chuyển đổi ảnh từ canvas sang định dạng phù hợp để dự đoán."""
    if canvas_result.image_data is None:
        return None
    img = canvas_result.image_data[:, :, :3]  # Chỉ lấy 3 kênh RGB
    img = Image.fromarray(img).convert("L").resize((28, 28))  # Chuyển sang grayscale, resize về 28x28
    img = np.array(img) / 255.0  # Chuẩn hóa về [0,1]
    img = img.reshape(1, -1)  # Đưa về dạng vector giống như trong `thi_nghiem()`
    return img

def du_doan():
    st.header("✍️ Vẽ số để dự đoán")

    # 📥 Load mô hình đã huấn luyện
    if "trained_model" in st.session_state:
        model = st.session_state["trained_model"]
        st.success("✅ Đã sử dụng mô hình vừa huấn luyện!")
    else:
        st.error("⚠️ Chưa có mô hình! Hãy huấn luyện trước.")


    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))  

    if st.button("🔄 Tải lại nếu không thấy canvas"):
        st.session_state.key_value = str(random.randint(0, 1000000))  

    # ✍️ Vẽ số
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,
        update_streamlit=True
    )

    if st.button("Dự đoán số"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            st.image(Image.fromarray((img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            # Dự đoán số
            prediction = model.predict(img)
            predicted_number = np.argmax(prediction, axis=1)[0]
            max_confidence = np.max(prediction)

            st.subheader(f"🔢 Dự đoán: {predicted_number}")
            st.write(f"📊 Mức độ tin cậy: {max_confidence:.2%}")

            # Hiển thị bảng confidence score
            prob_df = pd.DataFrame(prediction.reshape(1, -1), columns=[str(i) for i in range(10)]).T
            prob_df.columns = ["Mức độ tin cậy"]
            st.bar_chart(prob_df)

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")

    
from datetime import datetime    
import streamlit as st
import mlflow
from datetime import datetime

def show_experiment_selector():
    st.title("📊 MLflow")
    
    # Kết nối với DAGsHub MLflow Tracking
    mlflow.set_tracking_uri("https://dagshub.com/huydfdcv/my-first-repo.mlflow")
    
    # Lấy danh sách tất cả experiments
    experiment_name = "MNIST Pseudo Labelling"
    experiments = mlflow.search_experiments()
    selected_experiment = next((exp for exp in experiments if exp.name == experiment_name), None)

    if not selected_experiment:
        st.error(f"❌ Experiment '{experiment_name}' không tồn tại!")
        return

    st.subheader(f"📌 Experiment: {experiment_name}")
    st.write(f"**Experiment ID:** {selected_experiment.experiment_id}")
    st.write(f"**Trạng thái:** {'Active' if selected_experiment.lifecycle_stage == 'active' else 'Deleted'}")
    st.write(f"**Vị trí lưu trữ:** {selected_experiment.artifact_location}")

    # Lấy danh sách runs trong experiment
    runs = mlflow.search_runs(experiment_ids=[selected_experiment.experiment_id])

    if runs.empty:
        st.warning("⚠ Không có runs nào trong experiment này.")
        return

    st.write("### 🏃‍♂️ Các Runs gần đây:")
    
    # Lấy danh sách run_name từ params
    run_info = []
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        run_tags = mlflow.get_run(run_id).data.tags
        run_name = run_tags.get("mlflow.runName", f"Run {run_id[:8]}")  # Lấy từ tags
        run_info.append((run_name, run_id))
    
    # Tạo dictionary để map run_name -> run_id
    run_name_to_id = dict(run_info)
    run_names = list(run_name_to_id.keys())
    
    # Chọn run theo run_name
    selected_run_name = st.selectbox("🔍 Chọn một run:", run_names)
    selected_run_id = run_name_to_id[selected_run_name]

    # Hiển thị thông tin chi tiết của run được chọn
    selected_run = mlflow.get_run(selected_run_id)

    if selected_run:
        st.subheader(f"📌 Thông tin Run: {selected_run_name}")
        st.write(f"**Run ID:** {selected_run_id}")
        st.write(f"**Trạng thái:** {selected_run.info.status}")
        
        start_time_ms = selected_run.info.start_time  # Thời gian lưu dưới dạng milliseconds
        if start_time_ms:
            start_time = datetime.fromtimestamp(start_time_ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
        else:
            start_time = "Không có thông tin"
        
        st.write(f"**Thời gian chạy:** {start_time}")

        # Hiển thị thông số đã log
        params = selected_run.data.params
        metrics = selected_run.data.metrics

        if params:
            st.write("### ⚙️ Parameters:")
            st.json(params)

        if metrics:
            st.write("### 📊 Metrics:")
            st.json(metrics)

        # Kiểm tra và hiển thị dataset artifact
        dataset_path = f"{selected_experiment.artifact_location}/{selected_run_id}/artifacts/dataset.npy"
        st.write("### 📂 Dataset:")
        st.write(f"📥 [Tải dataset]({dataset_path})")
    else:
        st.warning("⚠ Không tìm thấy thông tin cho run này.")

def Neural_Network():
    #st.title("🚀 MLflow DAGsHub Tracking với Streamlit")
    
    if "mlflow_initialized" not in st.session_state:   
        DAGSHUB_MLFLOW_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
        st.session_state['mlflow_url']=DAGSHUB_MLFLOW_URI
        mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

        os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
        os.environ["MLFLOW_TRACKING_PASSWORD"] = "2CaXhRNYabm9fN3"
        st.session_state.mlflow_initialized = True
        mlflow.set_experiment("MNIST Pseudo Labelling")   
        
    
    
    # Tạo các tab với tiêu đề tương ứng
    tab1, tab2, tab3 = st.tabs([
        "🧠 Huấn luyện",
        "🖥️ DEMO",
        "🔥 MLflow"
    ])

    # Nội dung từng tab
    with tab1:
        st.title("🧠 Huấn luyện Neural Network trên MNIST")
        split_data()
        thi_nghiem()
    with tab2:
        du_doan()
    with tab3:
        show_experiment_selector()

if __name__ == "__main__":
    Neural_Network()