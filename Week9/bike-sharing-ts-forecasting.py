import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("bike-sharing-dataset.csv")
# print(data.info())

data["date_time"] = pd.to_datetime(data["date_time"])

# Hàm tạo time series data
def create_ts_data(data, window_size):
    for i in range(1, window_size + 1):
        data["users_{}".format(i)] = data["users"].shift(i)
    data = data.dropna(axis=0)
    return data

# Tạo time series data để dựa trên số lượng users của 24 giờ trước, dự đoán số lượng users của giờ thứ 25
window_size = 24
data = create_ts_data(data, window_size)

# Chọn các features vả target
x = data.drop(["date_time", "users"], axis=1)
y = data["users"]

# Chia bộ dữ liệu thành tập train và tập test
train_size = int(len(x) * 0.8)
x_train = x[:train_size]
x_test = x[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Tiền xử lý dữ liệu
num_features = ["temp", "atemp", "hum", "windspeed"] + ['users_{}'.format(i) for i in range(1, window_size + 1)]
cat_features = ["holiday", "workingday", "weather", "month", "hour", "weekday"]
preprocessor = ColumnTransformer(transformers=[
    ("num_features", StandardScaler(), num_features),
    ("cat_features", OneHotEncoder(sparse_output=False), cat_features)
])

# Xây dựng mô hình Linear Regression
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# Huấn luyện mô hình
model.fit(x_train, y_train)

# Dự đoán
y_predict = model.predict(x_test)
for i, j in zip(y_predict, y_test):
    print("Predicted value: {}. Actual value: {}".format(i, j))

# Đánh giá mô hình
print("MAE: {}".format(mean_absolute_error(y_test, y_predict)))
print("MSE: {}".format(mean_squared_error(y_test, y_predict)))
print("R2: {}".format(r2_score(y_test, y_predict)))

# Trực quang hóa dữ liệu
fig, ax = plt.subplots()
ax.plot(data["date_time"][:train_size], y_train.values, label="Train")
ax.plot(data["date_time"][train_size:], y_test.values, label="Actual")
ax.plot(data["date_time"][train_size:], y_predict, label="Predicted")
ax.set_xlabel("Time")
ax.set_ylabel("Users")
ax.legend()
ax.grid()
plt.show()



