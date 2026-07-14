# ==========================================
# TASK 1: IRIS FLOWER CLASSIFICATION
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -------------------------------
# Load Dataset
# -------------------------------
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
df["Species"] = df["Species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("="*50)
print("FIRST FIVE ROWS")
print(df.head())

# -------------------------------
# Exploratory Data Analysis
# -------------------------------
print("\nShape:", df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nNull Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

print("\nClass Distribution:")
print(df["Species"].value_counts())

# -------------------------------
# Pairplot
# -------------------------------
sns.pairplot(df, hue="Species")
plt.show()

# -------------------------------
# Boxplots
# -------------------------------
plt.figure(figsize=(12,8))

for i, column in enumerate(df.columns[:-1]):
    plt.subplot(2,2,i+1)
    sns.boxplot(x="Species", y=column, data=df)

plt.tight_layout()
plt.show()

# -------------------------------
# Feature Selection Discussion
# -------------------------------
print("\nFeature Selection Discussion:")
print("Petal Length and Petal Width are the most discriminative features.")
print("Sepal Length provides moderate separation.")
print("Sepal Width is the least discriminative because of overlapping classes.")

# -------------------------------
# Prepare Data
# -------------------------------
X = iris.data
y = iris.target

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# Logistic Regression
# ==========================================
print("\n" + "="*50)
print("LOGISTIC REGRESSION")

lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("Accuracy:", accuracy_lr)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_lr,
    target_names=iris.target_names
))

cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(5,4))
sns.heatmap(cm_lr, annot=True, cmap="Blues", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# K-Nearest Neighbours
# ==========================================
print("\n" + "="*50)
print("K-NEAREST NEIGHBOURS")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)

print("Accuracy:", accuracy_knn)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred_knn,
    target_names=iris.target_names
))

cm_knn = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(5,4))
sns.heatmap(cm_knn, annot=True, cmap="Greens", fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# Model Comparison
# ==========================================
results = pd.DataFrame({
    "Model": ["Logistic Regression", "KNN"],
    "Accuracy": [accuracy_lr, accuracy_knn]
})

print("\nModel Comparison")
print(results)

best_model = results.loc[results["Accuracy"].idxmax()]

print("\nBest Performing Model:")
print(best_model)

print("\nConclusion:")
print(f"The best-performing model is {best_model['Model']} with an accuracy of {best_model['Accuracy']:.2f}.")
print("Petal Length and Petal Width are the most important features for distinguishing the iris species.")