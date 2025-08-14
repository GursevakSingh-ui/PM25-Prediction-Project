import pandas as pd

# File paths (exact location of your datasets)
file_paths = {
    "Urban": r"F:\PRSA_Data_Dongsi.csv",
    "Suburban": r"F:\PRSA_Data_Shunyi.csv",
    "Rural": r"F:\PRSA_Data_Huairou.csv",
    "Industrial/Hotspot": r"F:\PRSA_Data_Gucheng.csv"
}

# List to store DataFrames
dfs = []

# Read and label each dataset
for category, path in file_paths.items():
    df = pd.read_csv(path)
    df['Category'] = category  # Add category column
    dfs.append(df)

# Merge all datasets
merged_df = pd.concat(dfs, ignore_index=True)

# Display merged dataset info
print("Merged Dataset Shape:", merged_df.shape)
print("\nColumns:", merged_df.columns.tolist())
print("\nSample Data:")
print(merged_df.head())

# Save merged dataset to F drive
output_path = r"F:\merged_data.csv"
merged_df.to_csv(output_path, index=False)
print(f"\nMerged dataset saved at: {output_path}")
