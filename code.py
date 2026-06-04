# Generated from: code.ipynb
# Converted at: 2026-06-04T12:30:51.832Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# **1. Import Libraries**



import pandas as pd
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier




# **2. Upload the Dataset**


dataset=pd.read_csv('dataset/customs_inspection_dataset_final_30000.csv')
dataset

# import random

# inspection_required = []

# for _, row in dataset.iterrows():

#     risk_score = 0

#     # =====================================================
#     # Commodity Risk Level (Highest Importance)
#     # =====================================================

#     if row["Commodity_Risk_Level"] == "High":
#         risk_score += 35

#     elif row["Commodity_Risk_Level"] == "Medium":
#         risk_score += 15

#     # =====================================================
#     # Previous Violations
#     # =====================================================

#     risk_score += min(
#         row["Previous_Violations"] * 4,
#         25
#     )

#     # =====================================================
#     # Missing Documents
#     # =====================================================

#     if row["Missing_Documents"] == "Yes":
#         risk_score += 20

#     # =====================================================
#     # Consignee Compliance Score
#     # =====================================================

#     if row["Consignee_Compliance_Score"] < 50:
#         risk_score += 20

#     elif row["Consignee_Compliance_Score"] < 70:
#         risk_score += 10

#     # =====================================================
#     # Shipper Compliance Score
#     # =====================================================

#     if row["Shipper_Compliance_Score"] < 50:
#         risk_score += 15

#     elif row["Shipper_Compliance_Score"] < 70:
#         risk_score += 8

#     # =====================================================
#     # Shipment Value
#     # =====================================================

#     if row["Shipment_Value_USD"] > 100000:
#         risk_score += 10

#     elif row["Shipment_Value_USD"] > 50000:
#         risk_score += 5

#     # =====================================================
#     # Cargo Season
#     # =====================================================

#     if row["Cargo_Season"] in [
#         "Ramadan",
#         "Hajj",
#         "Year_End_Peak"
#     ]:
#         risk_score += 5

#     # =====================================================
#     # Weight
#     # =====================================================

#     if row["Weight_KG"] > 5000:
#         risk_score += 3

#     # =====================================================
#     # Random Variation
#     # =====================================================

#     risk_score += random.randint(-5, 5)

#     # =====================================================
#     # Final Decision
#     # =====================================================

#     if risk_score >= 80:
#         inspection = 1
#     else:
#         inspection = 0

#     inspection_required.append(inspection)

# dataset["Inspection_Required"] = inspection_required

# **3. Basic Checkups**


# Target Distribution
dataset["Inspection_Required"].value_counts()

# Shape
print("Shape of the dataset:")
print(dataset.shape)
print("\n")


# Columns
print("Columns in the dataset:")
print(dataset.columns)
print("\n")

#check the null values in the dataset
print("Null values in the dataset:")
print(dataset.isnull().sum())

# Information about the dataset
print("Dataset Information:")
dataset.info()

#statistical summary of the dataset
print("Statistical Summary of the Dataset:")
dataset.describe()

# **4. Exploratory Data Analysis**


# 4.1 Target Variable Analysis


plt.figure(figsize=(8,5))

ax = sns.countplot(
    data=dataset,
    x='Inspection_Required'
)

plt.title('Customs Inspection Distribution')
plt.xlabel('Inspection Required')
plt.ylabel('Count')

# Show values above bars
for p in ax.patches:
    ax.annotate(
        f'{int(p.get_height())}',
        (p.get_x() + p.get_width()/2., p.get_height()),
        ha='center',
        va='bottom',
        fontsize=11,
        fontweight='bold'
    )

plt.tight_layout()
plt.show()

# 4.2 Commodity Type vs Inspection Required


plt.figure(figsize=(12,6))

ax = sns.countplot(
    data=dataset,
    x='Commodity_Type',
    hue='Inspection_Required'
)

total = len(dataset)

for p in ax.patches:
    height = p.get_height()

    if height > 0:
        percentage = 100 * height / total

        ax.annotate(
            f'{percentage:.1f}%',
            (p.get_x() + p.get_width()/2., height),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

plt.title('Commodity Type vs Inspection Required')
plt.xlabel('Commodity Type')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 4.3 Cargo Season vs Inspection Required


plt.figure(figsize=(10,6))

ax = sns.countplot(
    data=dataset,
    x='Cargo_Season',
    hue='Inspection_Required'
)

plt.title('Cargo Season vs Inspection Required')
plt.xlabel('Cargo Season')
plt.ylabel('Count')

for p in ax.patches:
    height = p.get_height()

    if height > 0:
        ax.annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width()/2., height),
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )

plt.tight_layout()
plt.show()

# 4.4 Missing Documents vs Inspection Required


plt.figure(figsize=(8,5))

ax = sns.countplot(
    data=dataset,
    x='Missing_Documents',
    hue='Inspection_Required'
)

plt.title('Missing Documents vs Inspection Required')
plt.xlabel('Missing Documents')
plt.ylabel('Count')

for p in ax.patches:
    height = p.get_height()

    if height > 0:
        ax.annotate(
            f'{int(height)}',
            (p.get_x() + p.get_width()/2., height),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

plt.tight_layout()
plt.show()

# **5. Feature Engineering**
# 


# 5.1 Dataset Spliting


drop_columns = ['Inspection_Required', 'Shipment_ID','Shipment_Date']

X = dataset.drop(drop_columns, axis=1)
y = dataset['Inspection_Required']

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Backup for CatBoost
X_train_cat = X_train.copy()
X_val_cat = X_val.copy()
X_test_cat = X_test.copy()

y_train_cat = y_train.copy()
y_val_cat = y_val.copy()    
y_test_cat = y_test.copy()

print("Train :", X_train.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# 5.2 Encoding the Categorical Features


#categorical value for encoding
categorical_cols = X_train.select_dtypes(include='object').columns
print(categorical_cols)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])
    X_test[col] = le.transform(X_test[col])
    
    label_encoders[col] = le

# 5.3 Standard Scaling for numeric Features


#numerical columns for Scaling
numerical_cols = X_train.select_dtypes(
    exclude='object'
).columns

numerical_cols


scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_val_scaled = X_val.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(
    X_train_scaled[numerical_cols]
)

X_val_scaled[numerical_cols] = scaler.transform(
    X_val_scaled[numerical_cols]
)

X_test_scaled[numerical_cols] = scaler.transform(
    X_test_scaled[numerical_cols]
)

# **6. Model Building**


# 6.1 Logistic Regression


###Grid Search for Logistic Regression###

# lr_param_grid = {
#     'C': [0.01, 0.1, 1, 10, 100],
#     'penalty': ['l1', 'l2'],
#     'solver': ['liblinear']
# }

# lr_grid = GridSearchCV(
#     estimator=LogisticRegression(
#         random_state=42,
#         max_iter=1000,
#         class_weight='balanced'
#     ),
#     param_grid=lr_param_grid,
#     cv=5,
#     scoring='f1',
#     n_jobs=-1
# )

# lr_grid.fit(X_train_scaled, y_train)

# print("Best Parameters:")
# print(lr_grid.best_params_)

# print("\nBest F1 Score:")
# print(lr_grid.best_score_)

lr_model = LogisticRegression(

    random_state=42,
    max_iter=1000,
    class_weight='balanced'
)

lr_model.fit(X_train_scaled,y_train)
y_prob_lr= lr_model.predict_proba(X_val_scaled)[:, 1]

threshold = 0.50

y_val_pred_lr = (y_prob_lr > threshold).astype(int)
print("Logistic Regression Model Trained Successfully")



# Accuracy
lr_accuracy = accuracy_score(y_val, y_val_pred_lr)

# Precision
lr_precision = precision_score(y_val, y_val_pred_lr)

# Recall
lr_recall = recall_score(y_val, y_val_pred_lr)

# F1 Score
lr_f1 = f1_score(y_val, y_val_pred_lr)

# Print metrics
print("Accuracy :", lr_accuracy)
print("Precision:", lr_precision)
print("Recall   :", lr_recall)
print("F1 Score :", lr_f1)

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_val_pred_lr))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred_lr))

# 6.2 Decision tree


###Grid Search for Decision Tree###

# dt_param_grid = {
#     'max_depth': [3, 5, 7, 10],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'criterion': ['gini', 'entropy']
# }

# dt_grid = GridSearchCV(
#     estimator=DecisionTreeClassifier(
#         random_state=42,
#         class_weight='balanced'
#     ),
#     param_grid=dt_param_grid,
#     cv=5,
#     scoring='f1',
#     n_jobs=-1
# )

# dt_grid.fit(X_train, y_train)

# print("Best Parameters:")
# print(dt_grid.best_params_)

# print("\nBest F1 Score:")
# print(dt_grid.best_score_)

# Initialize model
dt_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    class_weight='balanced'
)

# Train model
dt_model.fit(X_train, y_train)

# Validation prediction
y_prob_dt = dt_model.predict_proba(X_val)[:, 1]

# Custom threshold
threshold = 0.50

# Apply threshold
y_val_pred_dt = (y_prob_dt > threshold).astype(int)

print("Decision Tree Model Trained Successfully")

# Accuracy
dt_accuracy = accuracy_score(y_val, y_val_pred_dt)

# Precision
dt_precision = precision_score(y_val, y_val_pred_dt)

# Recall
dt_recall = recall_score(y_val, y_val_pred_dt)

# F1 Score
dt_f1 = f1_score(y_val, y_val_pred_dt)

# Print metrics
print("Accuracy :", dt_accuracy)
print("Precision:", dt_precision)
print("Recall   :", dt_recall)
print("F1 Score :", dt_f1)

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_val_pred_dt))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred_dt))

# 6.3 Random Forest


# ##Grid Search for Random Forest###

# rf_param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [5, 10, 15],
#     'min_samples_split': [2, 5],
#     'min_samples_leaf': [1, 2],
#     'max_features': ['sqrt']
# }

# rf_grid = GridSearchCV(
#     estimator=RandomForestClassifier(
#         random_state=42,
#         class_weight='balanced',
#         n_jobs=-1
#     ),
#     param_grid=rf_param_grid,
#     cv=5,
#     scoring='f1',
#     n_jobs=-1
# )

# rf_grid.fit(X_train, y_train)

# print("Best Parameters:")
# print(rf_grid.best_params_)

# print("\nBest F1 Score:")
# print(rf_grid.best_score_)

# Initialize model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced'
)

# Train model
rf_model.fit(X_train, y_train)

# Validation prediction
y_prob_rf = rf_model.predict_proba(X_val)[:, 1]

# Custom threshold
threshold = 0.45

# Apply threshold
y_val_pred_rf = (y_prob_rf > threshold).astype(int)

print("Random Forest Model Trained Successfully")

# Accuracy
rf_accuracy = accuracy_score(y_val, y_val_pred_rf)

# Precision
rf_precision = precision_score(y_val, y_val_pred_rf)

# Recall
rf_recall = recall_score(y_val, y_val_pred_rf)

# F1 Score
rf_f1 = f1_score(y_val, y_val_pred_rf)

# Print Metrics
print("Accuracy :", rf_accuracy)
print("Precision:", rf_precision)
print("Recall   :", rf_recall)
print("F1 Score :", rf_f1)

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_val_pred_rf))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred_rf))

# 6.4 XGBoost Classifier


###Random Search for XGBoost###

# xgb_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0],
#     'min_child_weight': [1, 3, 5]
# }

# xgb_random = RandomizedSearchCV(
#     estimator=XGBClassifier(
#         objective='binary:logistic',
#         eval_metric='logloss',
#         random_state=42
#     ),
#     param_distributions=xgb_param_grid,
#     n_iter=20,
#     scoring='f1',
#     cv=5,
#     verbose=1,
#     random_state=42,
#     n_jobs=-1
# )

# xgb_random.fit(X_train, y_train)

# print("Best Parameters:")
# print(xgb_random.best_params_)

# print("\nBest F1 Score:")
# print(xgb_random.best_score_)

# Tuned XGBoost with imbalance handling
xgb_model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=1.0,
    gamma=0.1,
    random_state=42,
)

# Train model
xgb_model.fit(X_train, y_train)

y_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Custom threshold
threshold = 0.50

# Apply threshold
y_val_pred_xgb = (y_prob_xgb > threshold).astype(int)

print("Tuned XGBoost Model Trained Successfully")

# Accuracy
xgb_accuracy = accuracy_score(y_val, y_val_pred_xgb)

# Precision
xgb_precision = precision_score(y_val, y_val_pred_xgb)

# Recall
xgb_recall = recall_score(y_val, y_val_pred_xgb)

# F1 Score
xgb_f1 = f1_score(y_val, y_val_pred_xgb)

# Print Metrics
print("Accuracy :", xgb_accuracy)
print("Precision:", xgb_precision)
print("Recall   :", xgb_recall)
print("F1 Score :", xgb_f1)

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_val_pred_xgb))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred_xgb))

# 6.5 LightGBM


# ##Random Search for LightGBM###

# lgbm_param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [3, 5, 7, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'num_leaves': [15, 31, 63],
#     'subsample': [0.8, 0.9, 1.0],
#     'colsample_bytree': [0.8, 0.9, 1.0]
# }

# lgbm_random = RandomizedSearchCV(
#     estimator=LGBMClassifier(
#         random_state=42,
#         class_weight='balanced',
#         verbose=-1
#     ),
#     param_distributions=lgbm_param_grid,
#     n_iter=20,
#     scoring='f1',
#     cv=5,
#     random_state=42,
#     n_jobs=-1
# )

# lgbm_random.fit(X_train, y_train)

# print("Best Parameters:")
# print(lgbm_random.best_params_)

# print("\nBest F1 Score:")
# print(lgbm_random.best_score_)

# Initialize model
lgbm_model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=10,
    num_leaves=15,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight='balanced',
    random_state=42
)
# Train model
lgbm_model.fit(X_train, y_train)

# Predict probabilities
y_prob_lgbm = lgbm_model.predict_proba(X_val)[:, 1]

# Custom threshold
threshold = 0.50

# Apply threshold
y_val_pred_lgbm = (y_prob_lgbm > threshold).astype(int)


print("LightGBM Model Trained Successfully")

# Accuracy
lgbm_accuracy = accuracy_score(y_val, y_val_pred_lgbm)

# Precision
lgbm_precision = precision_score(y_val, y_val_pred_lgbm)

# Recall
lgbm_recall = recall_score(y_val, y_val_pred_lgbm)

# F1 Score
lgbm_f1 = f1_score(y_val, y_val_pred_lgbm)

# Print Metrics
print("Accuracy :", lgbm_accuracy)
print("Precision:", lgbm_precision)
print("Recall   :", lgbm_recall)
print("F1 Score :", lgbm_f1)

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_val, y_val_pred_lgbm))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_val, y_val_pred_lgbm))

# 6.6 CATBoost Classifier


cat_features = X_train_cat.select_dtypes(
    include=['object', 'category']
).columns.tolist()


# # Random Search for CatBoost ###

# cat_param_grid = {
#     'iterations': [100, 200, 300],
#     'depth': [4, 6, 8, 10],
#     'learning_rate': [0.01, 0.05, 0.1, 0.2],
#     'l2_leaf_reg': [1, 3, 5, 7, 9]
# }

# cat_random = RandomizedSearchCV(
#     estimator=CatBoostClassifier(
#         random_state=42,
#         verbose=0
#     ),
#     param_distributions=cat_param_grid,
#     n_iter=20,
#     scoring='f1',
#     cv=5,
#     random_state=42,
#     n_jobs=-1
# )

# cat_random.fit(X_train_cat, y_train,cat_features=cat_features)

# print("Best Parameters:")
# print(cat_random.best_params_)

# print("\nBest F1 Score:")
# print(cat_random.best_score_)

# Initialize Model
cat_model = CatBoostClassifier(
    iterations=300,
    depth=10,
    l2_leaf_reg=7,
    learning_rate=0.05,
    loss_function='MultiClass',
    random_state=42,
    verbose=0
)

# Train Model
cat_model.fit(
    X_train_cat,
    y_train_cat,
    cat_features=cat_features
)

# Validation Prediction
y_val_pred_cat = cat_model.predict(X_val_cat)

# Convert prediction shape
y_val_pred_cat = y_val_pred_cat.flatten()

print("CatBoost Model Trained Successfully")

# Accuracy
cat_accuracy = accuracy_score(
    y_val_cat,
    y_val_pred_cat
)

# Precision
cat_precision = precision_score(
    y_val_cat,
    y_val_pred_cat
)

# Recall
cat_recall = recall_score(
    y_val_cat,
    y_val_pred_cat
)

# F1 Score
cat_f1 = f1_score(
    y_val_cat,
    y_val_pred_cat
)

print("Accuracy :", cat_accuracy)
print("Precision :", cat_precision)
print("Recall :", cat_recall)
print("F1 Score :", cat_f1)

print("\nConfusion Matrix:\n")
print(
    confusion_matrix(
        y_val_cat,
        y_val_pred_cat
    )
)

print("\nClassification Report:\n")
print(
    classification_report(
        y_val_cat,
        y_val_pred_cat
    )
)

# **7. Model Comparision**


# Create Model Comparison Table

model_results = pd.DataFrame({
    'Model': ['Logistic Regression','Decision Tree','Random Forest','XGBoost','LightGBM','CatBoost'],
    'Accuracy': [lr_accuracy,dt_accuracy,rf_accuracy,xgb_accuracy,lgbm_accuracy,cat_accuracy],
    'Precision': [lr_precision,dt_precision,rf_precision, xgb_precision,lgbm_precision,cat_precision],
    'Recall': [lr_recall, dt_recall, rf_recall, xgb_recall, lgbm_recall,cat_recall],
    'F1 Score': [lr_f1, dt_f1, rf_f1, xgb_f1,lgbm_f1,cat_f1]
})

# Sort by F1 Score
model_results = model_results.sort_values(
    by='F1 Score',
    ascending=False
)

# Display
print(model_results)

# **8. Save the Model**


import joblib

joblib.dump(
    cat_model,
    "model/customs_inspection_cat_model.pkl"
)
print("Model Saved Successfully")

joblib.dump(
    X_train.columns.tolist(),
    "model/feature_columns.pkl"
)
print("Feature Columns Saved")

joblib.dump(
    label_encoders,
    "model/label_encoders.pkl"
)
print("Label Encoders Saved")

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': cat_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance)


top_features = feature_importance.head(20)

plt.figure(figsize=(10,6))

ax = sns.barplot(
    data=top_features,
    x='Importance',
    y='Feature'
)

plt.title(
    'Top 10 Important Features - Random Forest',
    fontsize=14,
    fontweight='bold'
)

# Show values on bars
for i, value in enumerate(top_features['Importance']):
    plt.text(
        value,
        i,
        f'{value:.3f}',
        va='center',
        fontweight='bold'
    )

plt.tight_layout()
plt.show()