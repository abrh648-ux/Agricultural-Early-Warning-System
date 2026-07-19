# Machine Learning Pipeline — Step by Step Guide
## Agricultural Early Warning System for Ethiopia

---

## STEP 1: Data Collection
- Collect agricultural data (region, crop type, year, area, production, yield)
- Source: Ethiopian Ministry of Agriculture statistics (1996–2022)
- Format: Excel file (Etho-Agri Dataset-Enhanced.xlsx)

---

## STEP 2: Data Loading and Exploration
```python
import pandas as pd
import numpy as np

df = pd.read_excel("Etho-Agri Dataset-Enhanced.xlsx")

# Explore the data
print(df.shape)           # rows and columns
print(df.head())          # first 5 rows
print(df.info())          # column types
print(df.isnull().sum())  # missing values
print(df.describe())      # statistics
```

---

## STEP 3: Data Cleaning
```python
# Remove infinity values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing numeric values with median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Remove rows where both yield and production are zero
df = df[~((df['Yield (kg/ha)'] == 0) & (df['Production(kg)'] == 0))]
```

---

## STEP 4: Feature Engineering
Create new features to improve model accuracy.

```python
# Yield Growth Rate (year over year change)
df['Yield_Growth_Rate'] = df.groupby(['Region','crop type'])['Yield (kg/ha)'].pct_change()

# Production Growth Rate
df['Production_Growth_Rate'] = df.groupby(['Region','crop type'])['Production(kg)'].pct_change()

# Area Efficiency (production per hectare)
df['Area_Efficiency'] = df['Production(kg)'] / df['Area cultivated(Ha)'].replace(0, np.nan)

# Regional Average Yield
df['Regional_Avg_Yield'] = df.groupby('Region')['Yield (kg/ha)'].transform('mean')

# Crop Average Yield
df['Crop_Avg_Yield'] = df.groupby('crop type')['Yield (kg/ha)'].transform('mean')

# Yield Anomaly (how far from normal)
df['Yield_Anomaly'] = (
    df['Yield (kg/ha)'] - df['Regional_Avg_Yield']
) / df.groupby('Region')['Yield (kg/ha)'].transform('std')

# Rolling Yield Trend (3-year average)
df['Rolling_Yield_Trend'] = df.groupby(['Region','crop type'])['Yield (kg/ha)'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Yield Stability (rolling standard deviation)
df['Yield_Stability'] = df.groupby(['Region','crop type'])['Yield (kg/ha)'].transform(
    lambda x: x.rolling(3, min_periods=1).std()
)

# Production Area Ratio
df['Production_Area_Ratio'] = df['Production(kg)'] / df['Area cultivated(Ha)'].replace(0, np.nan)

# Early Warning Score (composite risk signal)
df['Early_Warning_Score'] = (
    -0.4 * df['Yield_Anomaly'].fillna(0) +
    -0.3 * df['Yield_Growth_Rate'].fillna(0) +
    -0.3 * df['Rolling_Yield_Trend'].fillna(0)
)

# Clean infinity values created by division
df.replace([np.inf, -np.inf], np.nan, inplace=True)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
```

---

## STEP 5: Encode Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder

le_region = LabelEncoder()
le_crop   = LabelEncoder()

df['Region_Code'] = le_region.fit_transform(df['Region'])
df['Crop_Code']   = le_crop.fit_transform(df['crop type'])
```

---

## STEP 6: Create Target Variable (Risk Label)
```python
def assign_risk_label(yield_anomaly):
    if yield_anomaly >= -0.5:
        return 0   # Low Risk
    elif yield_anomaly >= -1.5:
        return 1   # Medium Risk
    else:
        return 2   # High Risk

df['Risk_Label'] = df['Yield_Anomaly'].apply(assign_risk_label)

# Check distribution
print(df['Risk_Label'].value_counts())
# 0 = Low Risk
# 1 = Medium Risk
# 2 = High Risk
```

---

## STEP 7: Prepare Features and Target
```python
feature_cols = [
    'Region_Code', 'Crop_Code', 'Year',
    'Area cultivated(Ha)', 'Production(kg)',
    'Yield_Growth_Rate', 'Production_Growth_Rate',
    'Area_Efficiency', 'Regional_Avg_Yield', 'Crop_Avg_Yield',
    'Yield_Anomaly', 'Rolling_Yield_Trend', 'Yield_Stability',
    'Production_Area_Ratio', 'Early_Warning_Score'
]

X = df[feature_cols]
y = df['Risk_Label']
```

---

## STEP 8: Split Data into Train and Test
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # reproducibility
    stratify=y          # keep class balance
)

print("Train size:", X_train.shape)
print("Test size:",  X_test.shape)
```

---

## STEP 9: Handle Class Imbalance (Optional)
```python
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:",  y_train_bal.value_counts())
```

---

## STEP 10: Train Random Forest Model
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_bal, y_train_bal)
print("Random Forest training complete")
```

---

## STEP 11: Train XGBoost Model
```python
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)

xgb_model.fit(X_train_bal, y_train_bal)
print("XGBoost training complete")
```

---

## STEP 12: Evaluate Models
```python
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Predictions
y_pred_rf  = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Accuracy
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost Accuracy:",       accuracy_score(y_test, y_pred_xgb))

# Detailed report
print("\nRandom Forest Report:")
print(classification_report(y_test, y_pred_rf,
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))

print("\nXGBoost Report:")
print(classification_report(y_test, y_pred_xgb,
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))

# Confusion Matrix
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, model_name, y_pred in zip(
    axes,
    ['Random Forest', 'XGBoost'],
    [y_pred_rf, y_pred_xgb]
):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Low','Medium','High'],
                yticklabels=['Low','Medium','High'])
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
```

---

## STEP 13: SHAP Explainability
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb_model)

# Global feature importance
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction explanation
shap_single = explainer.shap_values(X_test.iloc[[0]])
explanation = shap.Explanation(
    values=shap_single[0][0],
    base_values=explainer.expected_value[0],
    data=X_test.iloc[0].values,
    feature_names=feature_cols
)
shap.plots.waterfall(explanation)
```

---

## STEP 14: Save Models and Data
```python
import joblib

# Save models
joblib.dump(xgb_model, "xgboost_model.pkl")
joblib.dump(rf_model,  "random_forest_model.pkl")

# Save test set for SHAP in the app
X_test.to_csv("X_test.csv", index=False)

# Save labeled dataset for the app
df.to_csv("labeled_agri_risk_data.csv", index=False)

# Save regional risk summary for the map
df['Predicted_Risk'] = xgb_model.predict(X)
regional = (
    df.groupby('Region')['Predicted_Risk']
    .mean().reset_index()
)
regional['Risk_Category'] = regional['Predicted_Risk'].apply(
    lambda x: 'High Risk' if x >= 1.5 else
              ('Medium Risk' if x >= 0.5 else 'Low Risk')
)
regional.to_csv("ethiopia_risk_map.csv", index=False)

print("All files saved successfully!")
print("Files created:")
print("  xgboost_model.pkl")
print("  random_forest_model.pkl")
print("  X_test.csv")
print("  labeled_agri_risk_data.csv")
print("  ethiopia_risk_map.csv")
```

---

## STEP 15: Deploy with Streamlit

Upload these files to GitHub:
```
your-repo/
├── app.py
├── requirements.txt
├── .python-version          (contains: 3.11)
├── xgboost_model.pkl
├── random_forest_model.pkl
├── X_test.csv
├── labeled_agri_risk_data.csv
├── ethiopia_risk_map.csv
└── ethiopia_regions.geojson
```

requirements.txt:
```
streamlit
joblib
matplotlib
numpy
pandas
plotly
scikit-learn
seaborn
shap==0.51.0
xgboost
```

Then go to share.streamlit.io → New app → select repo → main file: app.py → Deploy

---

## Summary Table

| Step | Action | Output |
|------|--------|--------|
| 1 | Collect data | Raw Excel file |
| 2 | Explore data | Understanding of structure |
| 3 | Clean data | No missing/infinity values |
| 4 | Feature engineering | 10 new features |
| 5 | Encode categories | Region_Code, Crop_Code |
| 6 | Create target variable | Risk_Label (0,1,2) |
| 7 | Prepare X and y | Feature matrix and labels |
| 8 | Train/test split | 80% train, 20% test |
| 9 | Handle imbalance | Balanced classes with SMOTE |
| 10 | Train Random Forest | rf_model |
| 11 | Train XGBoost | xgb_model |
| 12 | Evaluate | Accuracy, F1, Confusion Matrix |
| 13 | SHAP explainability | Feature importance charts |
| 14 | Save files | .pkl, .csv files |
| 15 | Deploy | Live Streamlit app |
