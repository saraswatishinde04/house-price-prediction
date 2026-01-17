import pandas as pd
import joblib
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
housing = fetch_california_housing(as_frame=True)
df = housing.frame
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/house_price_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Model and scaler saved successfully.")
