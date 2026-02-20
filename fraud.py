# =====================================================
# ONLINE PAYMENT FRAUD DETECTION - VS CODE VERSION
# ALL ALGORITHMS INCLUDED
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

print("Dataset Loaded")
print(df.head())

# =====================================================
# EDA GRAPHS
# =====================================================
sns.countplot(x="isFraud", data=df)
plt.title("Fraud vs Non-Fraud")
plt.show()

sns.countplot(x="type", data=df)
plt.xticks(rotation=45)
plt.title("Transaction Types")
plt.show()

sns.histplot(df["amount"], bins=50, kde=True)
plt.title("Transaction Amount Distribution")
plt.show()

# =====================================================
# DATA CLEANING
# =====================================================
df.drop(["nameOrig", "nameDest"], axis=1, inplace=True)

df["amount"] = np.log1p(df["amount"])

# =====================================================
# ENCODING
# =====================================================
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

pickle.dump(le, open("label_encoder.pkl","wb"))

# =====================================================
# SPLIT DATA
# =====================================================
X = df.drop("isFraud", axis=1)
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =====================================================
# HANDLE IMBALANCE
# =====================================================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:\n", y_train.value_counts())

# =====================================================
# SCALING FOR SVM
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pickle.dump(scaler, open("scaler.pkl","wb"))

# =====================================================
# CONFUSION MATRIX FUNCTION
# =====================================================
def plot_cm(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# =====================================================
# MODEL STORAGE
# =====================================================
results = []

# =====================================================
# DECISION TREE
# =====================================================
dt = DecisionTreeClassifier(class_weight="balanced",random_state=42)
dt.fit(X_train,y_train)
pred_dt = dt.predict(X_test)

plot_cm(y_test,pred_dt,"Decision Tree Confusion Matrix")

results.append([
    "Decision Tree",
    accuracy_score(y_test,pred_dt),
    precision_score(y_test,pred_dt),
    recall_score(y_test,pred_dt),
    f1_score(y_test,pred_dt),
    roc_auc_score(y_test,pred_dt)
])

# =====================================================
# RANDOM FOREST
# =====================================================
rf = RandomForestClassifier(n_estimators=100,class_weight="balanced",random_state=42)
rf.fit(X_train,y_train)
pred_rf = rf.predict(X_test)

plot_cm(y_test,pred_rf,"Random Forest Confusion Matrix")

results.append([
    "Random Forest",
    accuracy_score(y_test,pred_rf),
    precision_score(y_test,pred_rf),
    recall_score(y_test,pred_rf),
    f1_score(y_test,pred_rf),
    roc_auc_score(y_test,pred_rf)
])

# =====================================================
# EXTRA TREES
# =====================================================
et = ExtraTreesClassifier(n_estimators=100,random_state=42)
et.fit(X_train,y_train)
pred_et = et.predict(X_test)

plot_cm(y_test,pred_et,"Extra Trees Confusion Matrix")

results.append([
    "Extra Trees",
    accuracy_score(y_test,pred_et),
    precision_score(y_test,pred_et),
    recall_score(y_test,pred_et),
    f1_score(y_test,pred_et),
    roc_auc_score(y_test,pred_et)
])

# =====================================================
# SVM
# =====================================================
svm = SVC(kernel="rbf")

# faster training
X_small = pd.DataFrame(X_train).sample(50000,random_state=42)
y_small = y_train.loc[X_small.index]

svm.fit(scaler.transform(X_small),y_small)
pred_svm = svm.predict(X_test_scaled)

plot_cm(y_test,pred_svm,"SVM Confusion Matrix")

results.append([
    "SVM",
    accuracy_score(y_test,pred_svm),
    precision_score(y_test,pred_svm),
    recall_score(y_test,pred_svm),
    f1_score(y_test,pred_svm),
    roc_auc_score(y_test,pred_svm)
])

# =====================================================
# XGBOOST
# =====================================================
xgb = XGBClassifier(eval_metric="logloss",random_state=42)
xgb.fit(X_train,y_train)
pred_xgb = xgb.predict(X_test)

plot_cm(y_test,pred_xgb,"XGBoost Confusion Matrix")

results.append([
    "XGBoost",
    accuracy_score(y_test,pred_xgb),
    precision_score(y_test,pred_xgb),
    recall_score(y_test,pred_xgb),
    f1_score(y_test,pred_xgb),
    roc_auc_score(y_test,pred_xgb)
])

# =====================================================
# MODEL COMPARISON
# =====================================================
comparison = pd.DataFrame(
    results,
    columns=["Model","Accuracy","Precision","Recall","F1 Score","ROC AUC"]
)

print("\nMODEL COMPARISON")
print(comparison)

plt.figure(figsize=(10,5))
plt.bar(comparison["Model"],comparison["F1 Score"])
plt.title("Model Comparison (F1 Score)")
plt.xticks(rotation=45)
plt.show()

# =====================================================
# SAVE BEST MODEL
# =====================================================
best_name = comparison.sort_values(
    by="F1 Score",
    ascending=False
).iloc[0]["Model"]

models = {
    "Decision Tree":dt,
    "Random Forest":rf,
    "Extra Trees":et,
    "SVM":svm,
    "XGBoost":xgb
}

best_model = models[best_name]

pickle.dump(best_model,open("best_model.pkl","wb"))

print("Best Model:",best_name)
print("Model Saved Successfully")