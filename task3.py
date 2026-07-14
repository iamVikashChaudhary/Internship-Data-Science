# Car Price Prediction using Machine Learning

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------------- Load Dataset ----------------
df = pd.read_csv("car data.csv")

# ---------------- Data Cleaning ----------------
print("Shape:", df.shape)
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# Make categorical values consistent
df["Fuel_Type"] = df["Fuel_Type"].str.lower()
df["Seller_Type"] = df["Seller_Type"].str.lower()
df["Transmission"] = df["Transmission"].str.lower()

# ---------------- Feature Engineering ----------------
# Car Age
df["Car_Age"] = 2025 - df["Year"]

# Brand from Car Name
df["Brand"] = df["Car_Name"].str.split().str[0]

# ---------------- EDA ----------------

# Selling Price Distribution
sns.histplot(df["Selling_Price"], kde=True)
plt.title("Selling Price Distribution")
plt.show()

# Selling Price vs Fuel Type
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df)
plt.title("Selling Price vs Fuel Type")
plt.show()

# Selling Price vs Car Age
sns.scatterplot(x="Car_Age", y="Selling_Price", data=df)
plt.title("Selling Price vs Car Age")
plt.show()

# ---------------- Encoding ----------------
le = LabelEncoder()

for col in ["Fuel_Type","Seller_Type","Transmission","Brand"]:
    df[col] = le.fit_transform(df[col])

# ---------------- Heatmap ----------------
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ---------------- Features & Target ----------------
X = df.drop(["Car_Name","Selling_Price","Year"], axis=1)
y = df["Selling_Price"]

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Linear Regression ----------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

# ---------------- Random Forest ----------------
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)

# ---------------- Evaluation ----------------
def evaluate(name, y_test, pred):
    print(f"\n{name}")
    print("MAE :", mean_absolute_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R²  :", r2_score(y_test, pred))

evaluate("Linear Regression", y_test, pred_lr)
evaluate("Random Forest", y_test, pred_rf)

# ---------------- Best Model ----------------
if r2_score(y_test, pred_rf) > r2_score(y_test, pred_lr):
    print("\nBest Model: Random Forest")

    importance = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    importance.plot(kind="bar", figsize=(8,4))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.show()

else:
    print("\nBest Model: Linear Regression")