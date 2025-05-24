# analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Data
file_path = r"C:\Users\Panigrahi\Desktop\heart+disease\heart.csv"
df = pd.read_csv(file_path)
print("First 5 rows:\n", df.head())

# Step 2: Convert to NumPy Array
data = df.to_numpy()

# Step 3: Basic Stats with NumPy
age = df['age'].values
chol = df['chol'].values
thalach = df['thalach'].values  # Max heart rate
target = df['target'].values

print("Average Age:", np.mean(age))
print("Max Cholesterol:", np.max(chol))
print("Min Heart Rate:", np.min(thalach))
print("Cholesterol Std Dev:", np.std(chol))

# Step 4: Filter Patients with High Cholesterol
high_chol = df[df['chol'] > 240]
print("Patients with cholesterol > 240:", len(high_chol))

# Step 5: Correlation Between Age and Cholesterol
correlation = np.corrcoef(age, chol)[0, 1]
print("Age-Cholesterol Correlation:", correlation)

# Step 6: Target Distribution
unique, counts = np.unique(target, return_counts=True)
print("Heart Disease Presence Distribution:", dict(zip(unique, counts)))

# Step 7: Visualizations (optional)
plt.hist(age, bins=10, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.scatter(age, chol, c=target, cmap='coolwarm')
plt.title("Age vs Cholesterol")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.colorbar(label='Heart Disease')
plt.show()
