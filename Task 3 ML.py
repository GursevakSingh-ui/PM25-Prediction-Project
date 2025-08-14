import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Loading Dataset
file_path = r"F:/merged_data.csv"
df = pd.read_csv(file_path)

# ---------------------------
# 2. Preview Dataset
# ---------------------------
print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ---------------------------
# 3. Handle Missing Values
# ---------------------------
df = df.dropna()

# ---------------------------
# 4. Feature Selection
# ---------------------------
features = ["PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
target = "PM2.5"

X = df[features]
y = df[target]

# ---------------------------
# 5. Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# 6. Train Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# 7. Predictions
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 8. Evaluation
# ---------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:")
for feat, coef in zip(features, model.coef_):
    print(f"{feat}: {coef:.4f}")

print(f"\nIntercept: {model.intercept_:.4f}")
print(f"\nMean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f}")

# ---------------------------
# 9. Visualization
# ---------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("Actual vs Predicted PM2.5 Levels")
plt.grid(True)
plt.show()
