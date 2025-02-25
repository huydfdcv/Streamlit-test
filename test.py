import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Tải dữ liệu Titanic
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    st.write("10 dòng đầu của data trước khi xữ lý")
    st.write(df.head(10))   
    # Xóa các cột không cần thiết
    drop_colum = st.multiselect("Chọn các cột để xóa",['Name', 'Ticket', 'Cabin', 'PassengerId','Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'])
    if st.button("Xóa các cột"):
        df.drop(columns=drop_colum, inplace=True)
        st.success("Đã xóa các cộtcột")

    # Điền giá trị thiếu
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=False)

    # One-hot encoding cho cột danh mục
    if 'sex' not in drop_colum : df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    if 'Embarked' not in drop_colum : df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)

    return df

df = load_data()
st.title("🚢 Titanic Survival Prediction")
st.write("Dự đoán khả năng sống sót của hành khách trên Titanic bằng Hồi quy tuyến tính")

# Tạo tập biến đầu vào (X) và biến mục tiêu (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Chia tập dữ liệu thành Train (70%), Validation (15%) và Test (15%)
test_size = st.slider("Chọn tỷ lệ test:", 0.1, 0.5, 0.2)
valid_size = st.slider("Chọn tỷ lệ validation (trong tập test):", 0.1, 0.5, 0.2)
if st.button("Chia dữ liệu"):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=valid_size, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# Hàm huấn luyện mô hình
def train_model(model_type, degree=1):
    mlflow.set_experiment("Titanic_Regression")

    with mlflow.start_run():
        if model_type == "Polynomial Regression":
            poly = PolynomialFeatures(degree=degree)
            X_train_poly = poly.fit_transform(X_train_scaled)
            X_valid_poly = poly.transform(X_valid_scaled)
            model = LinearRegression()
            model.fit(X_train_poly, y_train)
            valid_pred = model.predict(X_valid_poly)
        else:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            valid_pred = model.predict(X_valid_scaled)

        valid_mse = mean_squared_error(y_valid, valid_pred)
        
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("validation_mse", valid_mse)
        mlflow.sklearn.log_model(model, "regression_model")

        return model, valid_mse

st.write("10 dòng đầu của data sau khi xữ lý")
st.write(df.head(10))

st.write("Tiền xữ lý dữ liệu:")
st.write("""
             - Các cột Name, ticket, cabin, passengerID sẽ bị xóa vì nó cản trở quá trình chuẩn hóa dữ liệu
             - Dữ liệu sẽ có những giá trị bị thiếu cần được xử lý:
              - **Age**: Các ô trống sẽ được fill bởi trung vị
              - **Embarked**: Các ô trống sẽ được fill bởi giá trị phổ biến nhất
            ```python
                    df['Age'].fillna(df['Age'].median(), inplace=True)
                df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
            ```
             """)
st.write("""
            - Biến đổi dữ liệu thành dạng số
            -    Embarked: Sử dụng 1 hot coding để chia thành 2 cột Embarked_Q, Embarked_S
            -    Sex_male: 1 cho male, 0 cho female
            ```python
                    df = pd.get_dummies(df, columns=['Embarked'], drop_first=False)
                    df = pd.get_dummies(df, columns=['Sex'], drop_first=TrueTrue)
            ```
             """) 
st.write("""
            -  Chuẩn hóa dữ liệu
             ```python
                    scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_valid_scaled = scaler.transform(X_valid)
                X_test_scaled = scaler.transform(X_test)
             ```
             """)


# Chọn mô hình
model_type = st.radio("Chọn mô hình:", ["Linear Regression", "Polynomial Regression"], key="model_selection")

# Nếu chọn Polynomial Regression, cho phép chọn bậc
degree = 2
if model_type == "Polynomial Regression":
    degree = st.slider("Chọn bậc của Polynomial Regression:", min_value=2, max_value=5, value=2, key="poly_degree")

# Huấn luyện mô hình
if st.button("Huấn luyện mô hình"):
    model, valid_mse = train_model(model_type, degree)
    st.success(f"Huấn luyện thành công! MSE trên tập Validation: {valid_mse:.4f}")

st.link_button(label= "MLflow",url = "http://127.0.0.1:5000/#/experiments/636393345947177791?viewStateShareKey=60b5838b3c07b10d688fd52b4dd6c37593b139dcfb12d21877e12fcb552682f6")
# Dự đoán kết quả với dữ liệu người dùng nhập vào
st.subheader("🔮 Dự đoán sống sót trên Titanic")

# Nhập dữ liệu hành khách mới
pclass = st.selectbox("Hạng vé (1: First, 2: Second, 3: Third)", [1, 2, 3])
age = st.slider("Tuổi", min_value=1, max_value=100, value=30)
fare = st.slider("Giá vé", min_value=0, max_value=500, value=50)
sibsp = st.slider("Số anh chị em / vợ chồng đi cùng", min_value=0, max_value=8, value=0)
parch = st.slider("Số cha mẹ / con cái đi cùng", min_value=0, max_value=6, value=0)
sex = st.radio("Giới tính", ["Nam", "Nữ"], key="gender")
embarked = st.radio("Nơi lên tàu", ["Q", "S", "C"], key="embarked")

# Chuyển đổi dữ liệu đầu vào
sex_male = 1 if sex == "Nam" else 0
embarked_S = 1 if embarked == "S" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_C = 1 if embarked == "C" else 0

# Tạo DataFrame từ dữ liệu đầu vào
input_data = pd.DataFrame([[pclass, age, sibsp, parch, fare, sex_male, embarked_C, embarked_Q, embarked_S]], 
                          columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S'])

# Chuẩn hóa dữ liệu đầu vào
input_scaled = scaler.transform(input_data)


# Dự đoán
if st.button("Dự đoán"):
    model, _ = train_model(model_type, degree)
    
    if model_type == "Polynomial Regression":
        poly = PolynomialFeatures(degree=degree)
        input_scaled_poly = poly.fit_transform(input_scaled)
        prediction = model.predict(input_scaled_poly)
    else:
        prediction = model.predict(input_scaled)

    survival_prob = prediction[0]
    st.write(f"🔮 Xác suất sống sót: **{survival_prob:.2f}**")

    if survival_prob > 0.5:
        st.success("✅ Hành khách này có khả năng sống sót!")
    else:
        st.error("❌ Hành khách này có khả năng không sống sót.")