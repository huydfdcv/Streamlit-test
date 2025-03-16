import streamlit as st
import mlflow
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import time

# Thiết lập MLflow với DagsHub
DAGSHUB_MLFLOW_URI = "https://dagshub.com/huydfdcv/my-first-repo.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "huydfdcv"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2CaXhRNYabm9fN3"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
mlflow.set_experiment("MNIST Pseudo Labelling")

# Hàm để tải dữ liệu MNIST
def load_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0
    return x_train, y_train, x_test, y_test

# Hàm để tạo mô hình Neural Network
def create_model():
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(28 * 28,)),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Hàm để lấy 1% dữ liệu từ mỗi class
def get_initial_data(x_train, y_train, percent=0.01):
    initial_x, initial_y = [], []
    for i in range(10):
        idx = np.where(y_train == i)[0]
        n_samples = int(len(idx) * percent)
        selected_idx = np.random.choice(idx, n_samples, replace=False)
        initial_x.append(x_train[selected_idx])
        initial_y.append(y_train[selected_idx])
    return np.concatenate(initial_x), np.concatenate(initial_y)

# Hàm để gán nhãn giả
def pseudo_labeling(model, x_unlabeled, threshold=0.95):
    probs = model.predict(x_unlabeled)
    pseudo_labels = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    mask = max_probs >= threshold
    return x_unlabeled[mask], pseudo_labels[mask]

# Hàm chính để huấn luyện và gán nhãn giả
def train_with_pseudo_labeling(x_train, y_train, x_test, y_test, threshold=0.95, max_iter=10, epochs=5):
    x_initial, y_initial = get_initial_data(x_train, y_train)
    x_unlabeled = np.delete(x_train, np.where(np.isin(x_train, x_initial).all(axis=1)), axis=0)
    
    model = create_model()
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001, step=0.0001)
    optimizer_name = st.sidebar.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"])
    
    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Thanh trạng thái cho quá trình huấn luyện
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for iteration in range(max_iter):
        with mlflow.start_run():
            # Huấn luyện mô hình
            for epoch in range(epochs):
                progress = (iteration * epochs + epoch + 1) / (max_iter * epochs)
                progress_bar.progress(progress)
                status_text.text(f"Iteration {iteration + 1}/{max_iter} - Epoch {epoch + 1}/{epochs}")
                model.fit(x_initial, y_initial, epochs=1, batch_size=32, validation_data=(x_test, y_test), verbose=0)
            
            # Gán nhãn giả
            x_pseudo, y_pseudo = pseudo_labeling(model, x_unlabeled, threshold)
            if len(x_pseudo) == 0:
                st.write(f"Iteration {iteration + 1}: No more pseudo labels to add.")
                break
            
            # Thêm dữ liệu được gán nhãn giả vào tập huấn luyện
            x_initial = np.concatenate([x_initial, x_pseudo])
            y_initial = np.concatenate([y_initial, y_pseudo])
            x_unlabeled = np.delete(x_unlabeled, np.where(np.isin(x_unlabeled, x_pseudo).all(axis=1)), axis=0)
            
            # Log thông tin
            mlflow.log_param("iteration", iteration + 1)
            mlflow.log_metric("train_size", len(x_initial))
            mlflow.log_metric("pseudo_labels_added", len(x_pseudo))
            
            # Đánh giá mô hình
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            mlflow.log_metric("test_accuracy", test_acc)
            st.write(f"Iteration {iteration + 1}: Test Accuracy = {test_acc:.4f}")
    
    return model

# Giao diện Streamlit
def main():
    st.title("Pseudo Labelling with Neural Network on MNIST")
    
    # Tải dữ liệu MNIST
    x_train, y_train, x_test, y_test = load_mnist()
    
    # Chia tập train/test
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # Hiển thị mẫu dữ liệu
    st.subheader("Sample Data from MNIST")
    num_samples = st.slider("Number of Samples to Display", 1, 10, 5)
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Label: {y_train[i]}")
        axes[i].axis('off')
    st.pyplot(fig)
    
    # Thiết lập các tham số
    threshold = st.sidebar.slider("Threshold for Pseudo Labeling", 0.8, 1.0, 0.95)
    max_iter = st.sidebar.slider("Maximum Iterations", 1, 20, 10)
    epochs = st.sidebar.slider("Epochs per Iteration", 1, 10, 5)
    
    # Bắt đầu huấn luyện
    if st.button("Start Training"):
        with mlflow.start_run():
            model = train_with_pseudo_labeling(x_train, y_train, x_test, y_test, threshold, max_iter, epochs)
            st.write("Training completed!")
            
            # Đánh giá mô hình cuối cùng
            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            st.subheader("Training Results")
            st.write(f"Final Test Accuracy: {test_acc:.4f}")
            st.write(f"Final Test Loss: {test_loss:.4f}")
            mlflow.log_metric("final_test_accuracy", test_acc)
            mlflow.log_metric("final_test_loss", test_loss)

if __name__ == "__main__":
    main()