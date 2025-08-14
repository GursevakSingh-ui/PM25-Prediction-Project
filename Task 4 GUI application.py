import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

DATASET_PATH = r"F:\merged_data.csv"
LIGHT_BG = "#f9f9f9"
ACCENT_COLOR = "#4CAF50"

# Load dataset
try:
    df = pd.read_csv(DATASET_PATH)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load dataset from {DATASET_PATH}:\n{e}")
    raise

root = tk.Tk()
root.title("Air Pollution Data Analysis")
root.geometry("1000x650")
root.config(bg=LIGHT_BG)

style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Arial", 12), padding=6)
style.configure("TLabel", background=LIGHT_BG, font=("Arial", 12))

frame_buttons = tk.Frame(root, bg=LIGHT_BG)
frame_buttons.pack(side="left", fill="y", padx=10, pady=10)

frame_content = tk.Frame(root, bg=LIGHT_BG)
frame_content.pack(side="right", expand=True, fill="both", padx=10, pady=10)

def show_data_overview():
    for widget in frame_content.winfo_children():
        widget.destroy()

    info_text = (
        f"Dataset Shape: {df.shape}\n\n"
        f"Columns:\n{', '.join(df.columns)}\n\n"
        f"Missing Values:\n{df.isnull().sum()}\n\n"
        f"First 5 rows:\n{df.head().to_string(index=False)}"
    )

    tk.Label(frame_content, text="Data Overview", font=("Arial", 16, "bold"), bg=LIGHT_BG, fg=ACCENT_COLOR).pack(pady=10)
    text_widget = tk.Text(frame_content, height=30, width=110, font=("Courier New", 10))
    text_widget.pack(pady=10)
    text_widget.insert(tk.END, info_text)
    text_widget.config(state='disabled')

def show_eda():
    for widget in frame_content.winfo_children():
        widget.destroy()

    tk.Label(frame_content, text="Exploratory Data Analysis", font=("Arial", 16, "bold"), bg=LIGHT_BG, fg=ACCENT_COLOR).pack(pady=10)

    fig, axs = plt.subplots(1, 2, figsize=(10,4), dpi=100)
    plt.tight_layout()

    # Histogram PM2.5
    if 'PM2.5' in df.columns:
        axs[0].hist(df['PM2.5'].dropna(), bins=30, color=ACCENT_COLOR, edgecolor='black')
        axs[0].set_title('Histogram of PM2.5')
        axs[0].set_xlabel('PM2.5')
        axs[0].set_ylabel('Frequency')
    else:
        axs[0].text(0.5, 0.5, 'PM2.5 not found', ha='center', va='center')

    # Scatter plot PM2.5 vs NO2
    if 'PM2.5' in df.columns and 'NO2' in df.columns:
        axs[1].scatter(df['NO2'], df['PM2.5'], alpha=0.5, color=ACCENT_COLOR)
        axs[1].set_title('PM2.5 vs NO2')
        axs[1].set_xlabel('NO2')
        axs[1].set_ylabel('PM2.5')
    else:
        axs[1].text(0.5, 0.5, 'NO2 or PM2.5 not found', ha='center', va='center')

    canvas = FigureCanvasTkAgg(fig, master=frame_content)
    canvas.draw()
    canvas.get_tk_widget().pack()

def show_modeling():
    for widget in frame_content.winfo_children():
        widget.destroy()

    tk.Label(frame_content, text="Modeling & Prediction", font=("Arial", 16, "bold"), bg=LIGHT_BG, fg=ACCENT_COLOR).pack(pady=10)

    # Prepare data for regression (predict PM2.5 using SO2, NO2, CO, O3, TEMP)
    features = ['SO2', 'NO2', 'CO', 'O3', 'TEMP']
    if not all(col in df.columns for col in features + ['PM2.5']):
        tk.Label(frame_content, text="Required columns for modeling not found.", bg=LIGHT_BG, fg="red").pack(pady=20)
        return

    data = df[features + ['PM2.5']].dropna()
    X = data[features]
    y = data['PM2.5']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results = (
        f"Linear Regression Model to predict PM2.5\n\n"
        f"Features: {', '.join(features)}\n\n"
        f"Mean Squared Error (MSE): {mse:.2f}\n"
        f"R^2 Score: {r2:.2f}\n\n"
        f"Model Coefficients:\n"
    )
    for feat, coef in zip(features, model.coef_):
        results += f"  {feat}: {coef:.4f}\n"
    results += f"Intercept: {model.intercept_:.4f}"

    text_widget = tk.Text(frame_content, height=20, width=90, font=("Courier New", 11))
    text_widget.pack(pady=20)
    text_widget.insert(tk.END, results)
    text_widget.config(state='disabled')

ttk.Button(frame_buttons, text="1. Data Overview", command=show_data_overview).pack(fill="x", pady=8)
ttk.Button(frame_buttons, text="2. Exploratory Data Analysis", command=show_eda).pack(fill="x", pady=8)
ttk.Button(frame_buttons, text="3. Modeling & Prediction", command=show_modeling).pack(fill="x", pady=8)

root.mainloop()
