# Agricultural Early Warning System
## Project Documentation — Full Pipeline from Data to Deployment

**Country:** Ethiopia  
**Domain:** Food Security / Agricultural Risk  
**Tech Stack:** Python, Scikit-learn, XGBoost, SHAP, Streamlit, GitHub  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Feature Engineering](#4-feature-engineering)
5. [Risk Label Creation (Target Variable)](#5-risk-label-creation-target-variable)
6. [Machine Learning Models](#6-machine-learning-models)
7. [Model Evaluation](#7-model-evaluation)
8. [Explainable AI (SHAP)](#8-explainable-ai-shap)
9. [Streamlit Web Application](#9-streamlit-web-application)
10. [Deployment on Streamlit Cloud](#10-deployment-on-streamlit-cloud)
11. [File Structure](#11-file-structure)
12. [Challenges and Solutions](#12-challenges-and-solutions)

---

## 1. Project Overview

The Agricultural Early Warning System is a machine learning-based web application
that predicts food security risk for Ethiopian regions. It allows users to select
a Year, Region, and Crop Type, and the system automatically predicts the risk level
(Low, Medium, or High) using trained XGBoost and Random Forest models.

The system also provides:
- Explainable AI (SHAP) to understand why a prediction was made
- Risk analysis heatmaps across all regions and crops
- A geographical risk map of Ethiopia

### Goal
Early detection of agricultural production risk to support food security
decision-making by government agencies, NGOs, and researchers.

### Risk Levels
| Level | Label | Color |
|-------|-------|-------|
| 0 | Low Risk | Green |
| 1 | Medium Risk | Orange |
| 2 | High Risk | Red |

---

## 2. Dataset Description

**Source:** Ethiopian agricultural production statistics  
**File:** `Etho-Agri Dataset-Enhanced.xlsx`  
**Processed File:** `labeled_agri_risk_data.csv`

### Coverage
- **Years:** 1996 – 2022 (2003 missing)
- **Regions:** 12 Ethiopian administrative regions
- **Crops:** 7 types — Teff, Barley, Wheat, Maize, Sorghum, Millet, Oats
- **Total Records:** ~1,807 rows

### Original Columns
| Column | Description |
|--------|-------------|
| Region | Ethiopian administrative region |
| crop type | Type of crop |
| Year | Production year |
| Area cultivated(Ha) | Area under cultivation in hectares |
| Production(kg) | Total production in kilograms |
| Yield (kg/ha) | Yield per hectare |
| Year_Since_Start | Years since 1996 |
| Decade | Decade grouping (1990, 2000, 2010, 2020) |
| Is_Recent | 1 if year >= 2010, else 0 |
| Region_Code | Numeric encoding of region |
| Region_Avg_Yield | Historical average yield for that region |
| Region_Growth_Rate | Year-over-year growth rate by region |
| Crop_Code | Numeric encoding of crop type |
| Crop_Avg_Yield | Historical average yield for that crop |
| Crop_Trend | Year-over-year crop yield trend |
| Area_Missing_Flag | 1 if area was originally missing |
| Area_Filled | Imputed area value |

### Data Quality Issues Found
- Many missing values in Area cultivated
- `inf` and `-inf` values in growth rate columns
- Zero production with non-zero area (conflict/drought years)
- Tigray 2022 all zeros (conflict impact)
- Yield = 0 when Production = 0

---

## 3. Data Preprocessing

### Steps Performed

**Step 1: Load and inspect data**
```python
import pandas as pd
df = pd.read_excel("Etho-Agri Dataset-Enhanced.xlsx")
df.shape        # Check dimensions
df.isnull().sum()  # Check missing values
df.dtypes       # Check column types
```

**Step 2: Handle infinity values**
```python
import numpy as np
df.replace([np.inf, -np.inf], np.nan, inplace=True)
```

**Step 3: Fill missing numeric values**
```python
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
```

**Step 4: Encode categorical columns**
```python
from sklearn.preprocessing import LabelEncoder
le_region = LabelEncoder()
le_crop   = LabelEncoder()
df['Region_Code'] = le_region.fit_transform(df['Region'])
df['Crop_Code']   = le_crop.fit_transform(df['crop type'])
```

**Step 5: Remove rows with zero yield where production is also zero**
```python
df = df[~((df['Yield (kg/ha)'] == 0) & (df['Production(kg)'] == 0))]
```

---

## 4. Feature Engineering

New features were created to capture agricultural patterns and trends.

### Engineered Features

| Feature | Formula | Purpose |
|---------|---------|---------|
| Yield_Growth_Rate | (yield_t - yield_t-1) / yield_t-1 | Year-over-year yield change |
| Production_Growth_Rate | (prod_t - prod_t-1) / prod_t-1 | Year-over-year production change |
| Area_Efficiency | Production / Area | Output per unit area |
| Regional_Avg_Yield | Mean yield per region (historical) | Regional baseline |
| Crop_Avg_Yield | Mean yield per crop (historical) | Crop baseline |
| Yield_Anomaly | (yield - Regional_Avg_Yield) / std | How far from normal |
| Rolling_Yield_Trend | 3-year rolling mean of yield | Smoothed trend |
| Yield_Stability | Rolling std of yield | Variability measure |
| Production_Area_Ratio | Production / Area (cleaned) | Efficiency ratio |
| Early_Warning_Score | Weighted combo of anomaly + trend | Composite risk signal |

### Code Example
```python
# Yield anomaly
df['Yield_Anomaly'] = (
    df['Yield (kg/ha)'] - df.groupby('Region')['Yield (kg/ha)'].transform('mean')
) / df.groupby('Region')['Yield (kg/ha)'].transform('std')

# Rolling yield trend (3-year window per region-crop group)
df['Rolling_Yield_Trend'] = (
    df.groupby(['Region', 'crop type'])['Yield (kg/ha)']
    .transform(lambda x: x.rolling(3, min_periods=1).mean())
)

# Early warning score
df['Early_Warning_Score'] = (
    -0.4 * df['Yield_Anomaly'] +
    -0.3 * df['Yield_Growth_Rate'] +
    -0.3 * df['Rolling_Yield_Trend']
)
```

---

## 5. Risk Label Creation (Target Variable)

The target variable `Risk_Label` was created based on yield deviation
from the regional historical average.

### Logic
```python
def assign_risk(row):
    anomaly = row['Yield_Anomaly']
    if anomaly >= -0.5:
        return 0   # Low Risk
    elif anomaly >= -1.5:
        return 1   # Medium Risk
    else:
        return 2   # High Risk

df['Risk_Label'] = df.apply(assign_risk, axis=1)
```

### Risk Distribution
The dataset was checked for class balance. If severely imbalanced,
SMOTE (Synthetic Minority Oversampling) was applied.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)
```

---

## 6. Machine Learning Models

### Features Used (Model Input)
```
Region_Code, Crop_Code, Year, Area cultivated(Ha), Production(kg),
Yield_Growth_Rate, Production_Growth_Rate, Area_Efficiency,
Regional_Avg_Yield, Crop_Avg_Yield, Yield_Anomaly,
Rolling_Yield_Trend, Yield_Stability, Production_Area_Ratio,
Early_Warning_Score
```

### Train/Test Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Model 1: Random Forest
```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)
```

### Model 2: XGBoost
```python
from xgboost import XGBClassifier
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)
```

### Save Models
```python
import joblib
joblib.dump(xgb_model,  "xgboost_model.pkl")
joblib.dump(rf_model,   "random_forest_model.pkl")

# Save test set for SHAP
X_test_df = pd.DataFrame(X_test, columns=feature_cols)
X_test_df.to_csv("X_test.csv", index=False)
```

---

## 7. Model Evaluation

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred_xgb = xgb_model.predict(X_test)
y_pred_rf  = rf_model.predict(X_test)

print("XGBoost:")
print(classification_report(y_test, y_pred_xgb,
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))

print("Random Forest:")
print(classification_report(y_test, y_pred_rf,
      target_names=['Low Risk', 'Medium Risk', 'High Risk']))
```

### Metrics Reported
- Accuracy
- Precision, Recall, F1-Score per class
- Confusion Matrix

### Risk Map Export
```python
# Generate regional risk summary
risk_map = df.copy()
risk_map['Predicted_Risk'] = xgb_model.predict(X)
regional = risk_map.groupby('Region')['Predicted_Risk'].mean().reset_index()
regional['Risk_Category'] = regional['Predicted_Risk'].apply(
    lambda x: 'High Risk' if x >= 1.5 else ('Medium Risk' if x >= 0.5 else 'Low Risk')
)
regional.to_csv("ethiopia_risk_map.csv", index=False)
```

---

## 8. Explainable AI (SHAP)

SHAP (SHapley Additive exPlanations) is used to explain both global
feature importance and individual predictions.

```python
import shap

explainer = shap.TreeExplainer(xgb_model)

# Global importance
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample, plot_type="bar")

# Individual prediction explanation (waterfall plot)
shap_single = explainer.shap_values(one_row)
explanation = shap.Explanation(
    values=shap_single[predicted_class][0],
    base_values=explainer.expected_value[predicted_class],
    data=one_row.values[0],
    feature_names=feature_cols
)
shap.plots.waterfall(explanation)
```

---

## 9. Streamlit Web Application

The app (`app.py`) has 7 pages:

| Page | Description |
|------|-------------|
| 🏠 Home | Overview and current prediction summary |
| 📂 Dataset | Explore raw data and statistics |
| 🤖 Early Warning Prediction | XGBoost + RF results, feature table, crop bar chart |
| 🔍 Explainable AI | SHAP waterfall for selected prediction + global importance |
| 📊 Risk Analysis | Heatmap and bar chart for all regions/crops |
| 🗺️ Ethiopia Risk Map | Choropleth map colored by predicted risk |
| ℹ️ About | Project information |

### User Inputs (Sidebar)
The user only provides 3 inputs:
1. **Year** — dropdown from 1996–2022
2. **Region** — dropdown of 12 Ethiopian regions
3. **Crop Type** — dropdown of 7 crops

All other features are automatically filled from historical data
using the closest matching year for that Region+Crop combination.

---

## 10. Deployment on Streamlit Cloud

### Files Required in GitHub Repository
```
agricultural-early-warning-system/
├── app.py                      ← Main Streamlit application
├── requirements.txt            ← Python dependencies
├── .python-version             ← Forces Python 3.11
├── labeled_agri_risk_data.csv  ← Processed dataset
├── X_test.csv                  ← Test features for SHAP
├── ethiopia_risk_map.csv       ← Regional risk summary
├── ethiopia_regions.geojson    ← Map boundaries
├── xgboost_model.pkl           ← Trained XGBoost model
└── random_forest_model.pkl     ← Trained Random Forest model
```

### requirements.txt
```
streamlit>=1.35.0
joblib>=1.3.0
matplotlib>=3.7.0
numpy>=1.26.0
pandas>=2.0.0
plotly>=5.18.0
scikit-learn>=1.3.0
seaborn>=0.12.0
shap==0.51.0
xgboost>=2.0.0
```

### .python-version
```
3.11
```

### Deployment Steps
1. Push all files to GitHub repository (main branch)
2. Go to share.streamlit.io
3. Click "New app"
4. Connect GitHub repository
5. Set main file as `app.py`
6. Click "Deploy"
7. Wait 3–8 minutes for build to complete

### Common Deployment Errors Fixed
| Error | Cause | Fix |
|-------|-------|-----|
| No response from server | streamlit missing from requirements.txt | Add streamlit to requirements.txt |
| shap install fails | shap==0.52.0 requires Python 3.12 | Use shap==0.51.0 with Python 3.11 |
| FileNotFoundError | CSV/PKL files missing from GitHub | Upload all data and model files |
| App crashes on load | st.set_option deprecation error | Remove deprecated option call |
| Force plot fails | JavaScript rendering unsupported | Replace with waterfall plot |

---

## 11. File Structure

```
d:\AgriRisk_ws\
├── app.py                          ← Final Streamlit app
├── requirements.txt                ← Deployment dependencies
├── .python-version                 ← Python 3.11 pin
├── warning_system.ipynb            ← Jupyter notebook (ML pipeline)
├── Etho-Agri Dataset-Enhanced.xlsx ← Original raw data
├── labeled_agri_risk_data.csv      ← Processed + labeled data
├── X_test.csv                      ← Test set features
├── ethiopia_risk_map.csv           ← Regional risk predictions
├── ethiopia_regions.geojson        ← Ethiopia region boundaries
├── xgboost_model.pkl               ← Saved XGBoost model
├── random_forest_model.pkl         ← Saved Random Forest model
└── Project_Documentation.md       ← This document
```

---

## 12. Challenges and Solutions

| Challenge | Solution |
|-----------|----------|
| Many missing values in area data | Used median imputation + Area_Missing_Flag |
| inf values in growth rate columns | Replaced with np.nan then median fill |
| Zero production years (conflict/drought) | Kept as valid signal for High Risk label |
| Class imbalance in risk labels | Applied SMOTE oversampling |
| SHAP force_plot not working in Streamlit | Replaced with shap.plots.waterfall |
| Python 3.14 incompatibility on Streamlit Cloud | Pinned to Python 3.11 via .python-version |
| shap==0.52.0 requires Python 3.12 | Downgraded to shap==0.51.0 |
| Users having to enter 15 features manually | Auto-fill from historical data using 3 inputs only |
| Duplicate title on Home page | Removed redundant st.title() call |

---

*Document prepared for: Agricultural Early Warning System Project*  
*Ethiopia — 1996 to 2022*  
*Models: XGBoost + Random Forest + SHAP Explainability*
