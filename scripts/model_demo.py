import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/cyber.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df = df.dropna()
df = df.drop_duplicates()

# -----------------------------
# Remove unnecessary columns
# -----------------------------
drop_cols = [
    'reported_year',
    'reported_month',
    'reported_day',
    'reported_hour'
]

df = df.drop(
    columns=[c for c in drop_cols if c in df.columns],
    errors='ignore'
)

# -----------------------------
# Encode categorical columns
# -----------------------------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------
# Create DEMO target variable
# -----------------------------
df['demo_target'] = (
    df['abuse_confidence_score'] > 99
).astype(int)

# -----------------------------
# Features and Target
# -----------------------------
X = df.drop('demo_target', axis=1)

# Keep only numeric columns
X = X.select_dtypes(include=['number'])

y = df['demo_target']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Model
# -----------------------------
model = DecisionTreeClassifier(
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:\n")

print(classification_report(
    y_test,
    y_pred
))

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:\n", cm)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()