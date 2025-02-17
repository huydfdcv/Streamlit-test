import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dữ liệu
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    return df

# Xữ lý dữ liệuliệu
def preprocess_data(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    return df

# Chia dữ liệu
def split_data(df):
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# Huấn luyện và tracking bằng MLFlow
def train_and_log_model(X_train, y_train, X_valid, y_valid):
    with mlflow.start_run():
        mlflow.set_tracking_uri('http://localhost:5000')
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Cross Validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        valid_acc = accuracy_score(y_valid, model.predict(X_valid))
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("cv_mean_accuracy", np.mean(cv_scores))
        mlflow.log_metric("validation_accuracy", valid_acc)
        
        mlflow.sklearn.log_model(model, "random_forest_model")
        return model

# Streamlit hiển thị kết quả
def main():
    st.title("Titanic Survival Prediction")
    df = load_data()
    st.write("10 dòng đầu của data trước khi xữ lý")
    st.write(df.head(10))
    df = preprocess_data(df)
    st.write("10 dòng đầu của data sau khi xữ lý")
    st.write(df.head(10))
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    
    st.write("Tiền xữ lý dữ liệu:")
    st.write("""
             Dữ liệu sẽ có những giá trị bị thiếu cần được xử lý:
              Age : Các ô trống sẽ được fill bởi median
              Embarked: Các ô trống sẽ được fill bởi mode
             """)
    if st.button("Train Model"):
        model = train_and_log_model(X_train, y_train, X_valid, y_valid)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        st.write(f"Độ chính xác: {acc:.4f}")
        st.text("Báo cáo phân loại:")
        st.text(report)
        
        st.success("Model đã được huấn luyện và log trêntrên MLFlow!")

if __name__ == "__main__":
    main()
