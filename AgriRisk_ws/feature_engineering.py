# =====================================================
# FEATURE ENGINEERING
# Project: Explainable AI-Based Agricultural Decision
#          Support System for Sustainable Farming in Ethiopia
# Dataset: Ethiopian Crop Production 1996-2022
# =====================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# -------------------------------------------------------
# STEP 1: LOAD DATA
# -------------------------------------------------------
df = pd.read_excel("Etho-Agri Dataset-Enhanced.xlsx")

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Standardize column names
df.columns = df.columns.str.strip()
df.rename(columns={
    "crop type":           "Crop_Type",
    "Area cultivated(Ha)": "Area_Ha",
    "Production(kg)":      "Production_Kg",
    "Yeild (kg/ha)":       "Yield_KgHa"
}, inplace=True)

# -------------------------------------------------------
# STEP 2: BASIC CLEANING
# -------------------------------------------------------

# Replace inf values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Convert numeric columns
num_cols = ["Area_Ha", "Production_Kg", "Yield_KgHa"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Fill missing area with region-crop median
df["Area_Ha"] = df.groupby(["Region", "Crop_Type"])["Area_Ha"] \
                  .transform(lambda x: x.fillna(x.median()))

# Flag rows where area was originally missing
df["Area_Missing_Flag"] = df["Area_Ha"].isnull().astype(int)

# Fill remaining missing with overall median
df["Area_Ha"].fillna(df["Area_Ha"].median(), inplace=True)
df["Production_Kg"].fillna(0, inplace=True)
df["Yield_KgHa"].fillna(0, inplace=True)

print("\nAfter cleaning shape:", df.shape)

# -------------------------------------------------------
# STEP 3: TIME FEATURES
# -------------------------------------------------------

df["Year_Since_Start"] = df["Year"] - df["Year"].min()

df["Decade"] = (df["Year"] // 10) * 10

df["Is_Recent"] = (df["Year"] >= 2010).astype(int)

# Season proxy (before/after 2005 policy shift)
df["Post_Policy"] = (df["Year"] >= 2005).astype(int)

print("Time features added.")

# -------------------------------------------------------
# STEP 4: YIELD-BASED FEATURES
# -------------------------------------------------------

# 4.1 Regional average yield (historical baseline)
df["Regional_Avg_Yield"] = df.groupby("Region")["Yield_KgHa"] \
                             .transform("mean")

# 4.2 Crop average yield (crop baseline)
df["Crop_Avg_Yield"] = df.groupby("Crop_Type")["Yield_KgHa"] \
                         .transform("mean")

# 4.3 Regional-Crop average yield (combined baseline)
df["Region_Crop_Avg_Yield"] = df.groupby(["Region", "Crop_Type"])["Yield_KgHa"] \
                                 .transform("mean")

# 4.4 Yield anomaly — how far this yield is from regional average
df["Regional_Yield_Std"] = df.groupby("Region")["Yield_KgHa"] \
                             .transform("std").replace(0, np.nan)
df["Yield_Anomaly"] = (
    (df["Yield_KgHa"] - df["Regional_Avg_Yield"]) /
    df["Regional_Yield_Std"]
)
df["Yield_Anomaly"].fillna(0, inplace=True)

# 4.5 Yield deviation from crop baseline
df["Yield_Deviation_From_Crop"] = df["Yield_KgHa"] - df["Crop_Avg_Yield"]

# 4.6 Yield ratio — current yield vs regional average
df["Yield_Ratio"] = df["Yield_KgHa"] / df["Regional_Avg_Yield"].replace(0, np.nan)
df["Yield_Ratio"].fillna(1, inplace=True)

print("Yield features added.")

# -------------------------------------------------------
# STEP 5: GROWTH RATE FEATURES
# -------------------------------------------------------

df = df.sort_values(["Region", "Crop_Type", "Year"])

# 5.1 Year-over-year yield growth rate
df["Yield_Growth_Rate"] = df.groupby(["Region", "Crop_Type"])["Yield_KgHa"] \
    .pct_change()

# 5.2 Year-over-year production growth rate
df["Production_Growth_Rate"] = df.groupby(["Region", "Crop_Type"])["Production_Kg"] \
    .pct_change()

# 5.3 Year-over-year area growth rate
df["Area_Growth_Rate"] = df.groupby(["Region", "Crop_Type"])["Area_Ha"] \
    .pct_change()

# Replace inf from pct_change on zero values
df[["Yield_Growth_Rate",
    "Production_Growth_Rate",
    "Area_Growth_Rate"]] = \
    df[["Yield_Growth_Rate",
        "Production_Growth_Rate",
        "Area_Growth_Rate"]].replace([np.inf, -np.inf], np.nan).fillna(0)

print("Growth rate features added.")

# -------------------------------------------------------
# STEP 6: ROLLING / TREND FEATURES
# -------------------------------------------------------

# 6.1 3-year rolling average yield (smoothed trend)
df["Rolling_Yield_3yr"] = df.groupby(["Region", "Crop_Type"])["Yield_KgHa"] \
    .transform(lambda x: x.rolling(3, min_periods=1).mean())

# 6.2 5-year rolling average yield
df["Rolling_Yield_5yr"] = df.groupby(["Region", "Crop_Type"])["Yield_KgHa"] \
    .transform(lambda x: x.rolling(5, min_periods=1).mean())

# 6.3 Yield stability — rolling standard deviation (lower = more stable)
df["Yield_Stability"] = df.groupby(["Region", "Crop_Type"])["Yield_KgHa"] \
    .transform(lambda x: x.rolling(3, min_periods=1).std()).fillna(0)

# 6.4 Rolling production trend
df["Rolling_Production_3yr"] = df.groupby(["Region", "Crop_Type"])["Production_Kg"] \
    .transform(lambda x: x.rolling(3, min_periods=1).mean())

# 6.5 Trend direction — is yield increasing or decreasing?
df["Yield_Trend_Direction"] = np.sign(df["Yield_Growth_Rate"])

print("Rolling/trend features added.")

# -------------------------------------------------------
# STEP 7: EFFICIENCY FEATURES
# -------------------------------------------------------

# 7.1 Area efficiency — production per hectare
df["Area_Efficiency"] = df["Production_Kg"] / df["Area_Ha"].replace(0, np.nan)
df["Area_Efficiency"].fillna(0, inplace=True)

# 7.2 Production-area ratio (normalized)
df["Production_Area_Ratio"] = df["Production_Kg"] / \
    (df["Area_Ha"] * df["Regional_Avg_Yield"]).replace(0, np.nan)
df["Production_Area_Ratio"].fillna(0, inplace=True)

# 7.3 Relative efficiency vs region average
df["Regional_Efficiency_Avg"] = df.groupby("Region")["Area_Efficiency"] \
                                   .transform("mean")
df["Relative_Efficiency"] = df["Area_Efficiency"] / \
    df["Regional_Efficiency_Avg"].replace(0, np.nan)
df["Relative_Efficiency"].fillna(1, inplace=True)

print("Efficiency features added.")

# -------------------------------------------------------
# STEP 8: REGIONAL CONTEXT FEATURES
# -------------------------------------------------------

# 8.1 Region's share of national production for that year
df["National_Production_Year"] = df.groupby("Year")["Production_Kg"] \
                                    .transform("sum")
df["Region_Production_Share"] = df["Production_Kg"] / \
    df["National_Production_Year"].replace(0, np.nan)
df["Region_Production_Share"].fillna(0, inplace=True)

# 8.2 Crop's share of regional production for that year
df["Region_Year_Production"] = df.groupby(["Region", "Year"])["Production_Kg"] \
                                   .transform("sum")
df["Crop_Share_In_Region"] = df["Production_Kg"] / \
    df["Region_Year_Production"].replace(0, np.nan)
df["Crop_Share_In_Region"].fillna(0, inplace=True)

# 8.3 Number of years this region-crop has been productive
df["Productive_Years"] = df.groupby(["Region", "Crop_Type"])["Yield_KgHa"] \
    .transform(lambda x: (x > 0).cumsum())

print("Regional context features added.")

# -------------------------------------------------------
# STEP 9: EARLY WARNING SCORE (composite risk signal)
# -------------------------------------------------------

# Normalize components to 0-1 scale for the score
def minmax(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

# Higher anomaly (negative) = higher risk
risk_anomaly    = minmax(-df["Yield_Anomaly"].clip(-3, 3))
# Negative growth = higher risk
risk_growth     = minmax(-df["Yield_Growth_Rate"].clip(-2, 2))
# Lower rolling trend = higher risk
risk_trend      = minmax(-df["Rolling_Yield_3yr"])
# Lower stability (more volatile) = higher risk
risk_stability  = minmax(df["Yield_Stability"])

# Weighted composite score
df["Early_Warning_Score"] = (
    0.35 * risk_anomaly +
    0.25 * risk_growth  +
    0.25 * risk_trend   +
    0.15 * risk_stability
)

print("Early Warning Score added.")

# -------------------------------------------------------
# STEP 10: ENCODE CATEGORICAL FEATURES
# -------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

le_region = LabelEncoder()
le_crop   = LabelEncoder()

df["Region_Code"] = le_region.fit_transform(df["Region"])
df["Crop_Code"]   = le_crop.fit_transform(df["Crop_Type"])

# Save encoder mappings for reference
region_mapping = dict(zip(le_region.classes_,
                           le_region.transform(le_region.classes_)))
crop_mapping   = dict(zip(le_crop.classes_,
                           le_crop.transform(le_crop.classes_)))
print("\nRegion encoding:", region_mapping)
print("Crop encoding:  ", crop_mapping)

# -------------------------------------------------------
# STEP 11: CREATE RISK LABEL (Target Variable)
# -------------------------------------------------------

def assign_risk(row):
    """
    Risk based on yield anomaly (deviation from regional average):
    Low Risk    : yield >= -0.5 std from mean
    Medium Risk : yield between -0.5 and -1.5 std
    High Risk   : yield < -1.5 std from mean
    """
    anomaly = row["Yield_Anomaly"]
    if anomaly >= -0.5:
        return 0   # Low Risk
    elif anomaly >= -1.5:
        return 1   # Medium Risk
    else:
        return 2   # High Risk

df["Risk_Label"] = df.apply(assign_risk, axis=1)

risk_dist = df["Risk_Label"].value_counts()
print("\nRisk label distribution:")
print(f"  Low Risk    (0): {risk_dist.get(0, 0)}")
print(f"  Medium Risk (1): {risk_dist.get(1, 0)}")
print(f"  High Risk   (2): {risk_dist.get(2, 0)}")

# -------------------------------------------------------
# STEP 12: FINAL FEATURE LIST
# -------------------------------------------------------

FINAL_FEATURES = [
    # Identifiers
    "Region_Code", "Crop_Code", "Year",

    # Raw inputs
    "Area_Ha", "Production_Kg",

    # Time features
    "Year_Since_Start", "Is_Recent", "Post_Policy",

    # Yield features
    "Yield_Anomaly", "Yield_Ratio", "Yield_Deviation_From_Crop",
    "Regional_Avg_Yield", "Crop_Avg_Yield", "Region_Crop_Avg_Yield",

    # Growth features
    "Yield_Growth_Rate", "Production_Growth_Rate", "Area_Growth_Rate",
    "Yield_Trend_Direction",

    # Rolling/trend features
    "Rolling_Yield_3yr", "Rolling_Yield_5yr",
    "Yield_Stability", "Rolling_Production_3yr",

    # Efficiency features
    "Area_Efficiency", "Production_Area_Ratio", "Relative_Efficiency",

    # Regional context
    "Region_Production_Share", "Crop_Share_In_Region", "Productive_Years",

    # Composite score
    "Early_Warning_Score",

    # Flags
    "Area_Missing_Flag",
]

TARGET = "Risk_Label"

print(f"\nTotal features: {len(FINAL_FEATURES)}")
print(f"Target: {TARGET}")

# -------------------------------------------------------
# STEP 13: CLEAN FINAL DATASET
# -------------------------------------------------------

# Keep only needed columns
df_final = df[FINAL_FEATURES + [TARGET, "Region", "Crop_Type", "Year"]].copy()

# Final cleanup of any remaining inf/nan
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in FINAL_FEATURES:
    if df_final[col].isnull().any():
        df_final[col].fillna(df_final[col].median(), inplace=True)

# Remove rows where yield was zero (no data / not cultivated)
df_final = df_final[df_final["Production_Kg"] > 0].reset_index(drop=True)

print(f"\nFinal dataset shape: {df_final.shape}")
print(f"Missing values: {df_final.isnull().sum().sum()}")

# -------------------------------------------------------
# STEP 14: SAVE
# -------------------------------------------------------

df_final.to_csv("labeled_agri_risk_data.csv", index=False)
print("\nSaved: labeled_agri_risk_data.csv")

# Show sample
print("\nSample rows:")
print(df_final.head(3).to_string())

# Feature summary
print("\nFeature summary:")
print(df_final[FINAL_FEATURES].describe().T[["mean","std","min","max"]].round(2))
