# Unemployment Analysis in India

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")
df.columns = df.columns.str.strip()

# Data Inspection
print("Shape:", df.shape)
print(df.info())
print(df.isnull().sum())

# Type Conversion
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# ------------------- EDA -------------------

# Region-wise Average Unemployment Rate
region_avg = df.groupby("Region")["Estimated Unemployment Rate (%)"].mean().sort_values()
print(region_avg)

# Month-wise Trend
df["Month"] = df["Date"].dt.month_name()
month_avg = df.groupby("Month")["Estimated Unemployment Rate (%)"].mean()

# ------------------- Line Chart -------------------

states = ["Delhi", "Maharashtra", "Tamil Nadu"]

plt.figure(figsize=(10,5))
for state in states:
    data = df[df["Region"] == state]
    plt.plot(data["Date"], data["Estimated Unemployment Rate (%)"], label=state)

plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.legend()
plt.show()

# Observation:
# Compare unemployment trends of Delhi, Maharashtra and Tamil Nadu over time.

# ------------------- Bar Chart -------------------

top10 = region_avg.sort_values(ascending=False).head(10)

plt.figure(figsize=(8,5))
top10.plot(kind="bar", color="skyblue")
plt.title("Top 10 States with Highest Average Unemployment")
plt.ylabel("Average Unemployment Rate (%)")
plt.show()

# Observation:
# Shows the states with the highest average unemployment rates.

# ------------------- Heatmap -------------------

corr = df[[
    "Estimated Unemployment Rate (%)",
    "Estimated Employed",
    "Estimated Labour Participation Rate (%)"
]].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Observation:
# Displays correlation between unemployment, employment and labour participation.

# ------------------- Pre vs Post COVID -------------------

pre = df[df["Date"] < "2020-03-25"]
post = df[df["Date"] >= "2020-03-25"]

print("\nPre-COVID Average Unemployment:",
      pre["Estimated Unemployment Rate (%)"].mean())

print("Post-COVID Average Unemployment:",
      post["Estimated Unemployment Rate (%)"].mean())

# Observation:
# Compare the average unemployment rate before and after COVID lockdown.