import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the dataset
file_path = r"F:\merged_data.csv"
df = pd.read_csv(file_path)

print("=== FUNDAMENTAL DATA UNDERSTANDING ===\n")
# Shape of dataset
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

# Columns and data types
print("Data Types:\n", df.dtypes, "\n")

# Missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values[missing_values > 0], "\n")

# Unique values per column
print("Unique Values Count:\n", df.nunique(), "\n")

# Quick sample
print("First 5 Rows:\n", df.head(), "\n")


# =======================
# DATA PREPROCESSING
# =======================
print("=== DATA PREPROCESSING ===\n")

# Remove duplicate rows
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"Removed {initial_rows - df.shape[0]} duplicate rows.\n")

# Handle missing values:
# Example: Fill numeric NaNs with median, categorical NaNs with mode
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
cat_cols = df.select_dtypes(include=['object']).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after filling:\n", df.isnull().sum().sum(), " total\n")

# Feature Engineering: create datetime column
df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
print("Added 'datetime' column.\n")


# =======================
# STATISTICAL SUMMARY
# =======================
print("=== STATISTICAL SUMMARY ===\n")
print(df.describe().T, "\n")


# =======================
# VISUALISATIONS
# =======================
sns.set(style="whitegrid")
plt.figure(figsize=(10, 5))
sns.histplot(df['PM2.5'], bins=50, kde=True)
plt.title('PM2.5 Distribution')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='Category', y='PM2.5', data=df)
plt.title('PM2.5 by Location Category')
plt.show()

plt.figure(figsize=(10, 5))
sns.scatterplot(x='TEMP', y='PM2.5', hue='Category', data=df)
plt.title('Temperature vs PM2.5')
plt.show()

plt.figure(figsize=(10, 5))
sns.lineplot(x='datetime', y='PM2.5', hue='Category', data=df)
plt.title('PM2.5 Trends Over Time')
plt.show()

plt.figure(figsize=(8, 6))
corr = df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

print("=== EDA Completed ===\n")
