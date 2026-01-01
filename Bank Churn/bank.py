# ===============================
# 1. IMPORT LIBRARIES
# ===============================

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # <-- Prevents Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# ===============================
# 2. LOAD DATA
# ===============================

df = pd.read_csv("Churn_Modelling.csv")

# Preview the data
print(df.head())

# ===============================
# 3. BASIC EDA
# ===============================

# Check dataset info
print(df.info())

# Churn distribution
sns.countplot(x="churn", data=df)
plt.title("Customer Churn Distribution")
plt.savefig("churn_countplot.png")  # Save plot instead of showing
plt.close()

# Balance vs churn
sns.boxplot(x="churn", y="balance", data=df)
plt.title("Balance vs Churn")
plt.savefig("balance_vs_churn.png")
plt.close()

# ===============================
# 4. DATA CLEANING
# ===============================

# Drop columns that don't help prediction
df = df.drop(columns=["customer_id"])

# Encode categorical variables
le = LabelEncoder()
df["gender"] = le.fit_transform(df["gender"])
df["country"] = le.fit_transform(df["country"])

# ===============================
# 5. FEATURES & TARGET
# ===============================

X = df.drop("churn", axis=1)
y = df["churn"]

# ===============================
# 6. TRAIN / TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ===============================
# 7. FEATURE SCALING
# ===============================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 8. MODEL TRAINING
# ===============================

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ===============================
# 9. MODEL EVALUATION
# ===============================

# Predictions
y_pred = model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

print("Plots saved as PNG files in your project folder.")
