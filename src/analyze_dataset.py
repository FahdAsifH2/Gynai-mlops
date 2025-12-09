import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv("maternal_data_cleaned_encoded.csv")

print("="*80)
print("COMPLETE DATASET ANALYSIS REPORT")
print("="*80)

# ============================================================================
# 1. BASIC INFORMATION
# ============================================================================
print("\n" + "="*80)
print("1. BASIC DATASET INFORMATION")
print("="*80)
print(f"\nDataset Shape: {df.shape}")
print(f"Number of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")
print(f"\nColumn Names:\n{list(df.columns)}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMemory Usage:\n{df.memory_usage(deep=True)}")
print(f"\nTotal Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# ============================================================================
# 2. MISSING VALUES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("2. MISSING VALUES ANALYSIS")
print("="*80)
missing = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_percent.values
})
print(f"\nTotal Missing Values: {df.isnull().sum().sum()}")
print(f"\nMissing Values Per Column:")
print(missing_df[missing_df['Missing_Count'] > 0])
if df.isnull().sum().sum() == 0:
    print("\n✅ NO MISSING VALUES FOUND!")

# ============================================================================
# 3. DUPLICATE ROWS
# ============================================================================
print("\n" + "="*80)
print("3. DUPLICATE ROWS ANALYSIS")
print("="*80)
duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicate Rows: {duplicates}")
if duplicates > 0:
    print(f"Percentage of Duplicates: {(duplicates/len(df))*100:.2f}%")
else:
    print("✅ NO DUPLICATES FOUND!")

# ============================================================================
# 4. TARGET VARIABLE ANALYSIS (CLASS IMBALANCE)
# ============================================================================
print("\n" + "="*80)
print("4. TARGET VARIABLE ANALYSIS - Delivery_Category_Encoded")
print("="*80)
target_col = "Delivery_Category_Encoded"
print(f"\nClass Distribution (Count):")
print(df[target_col].value_counts())
print(f"\nClass Distribution (Percentage):")
print(df[target_col].value_counts(normalize=True) * 100)

class_counts = df[target_col].value_counts()
majority_class = class_counts.max()
minority_class = class_counts.min()
imbalance_ratio = majority_class / minority_class

print(f"\n📊 Class Imbalance Metrics:")
print(f"Majority Class Count: {majority_class}")
print(f"Minority Class Count: {minority_class}")
print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print(f"⚠️ DATASET IS IMBALANCED! Consider using SMOTE.")
else:
    print(f"✅ Dataset is relatively balanced.")

# ============================================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("5. DESCRIPTIVE STATISTICS FOR ALL FEATURES")
print("="*80)
print(f"\n{df.describe(include='all').T}")

# ============================================================================
# 6. DETAILED STATISTICS PER COLUMN
# ============================================================================
print("\n" + "="*80)
print("6. DETAILED STATISTICS PER COLUMN")
print("="*80)

for col in df.columns:
    print(f"\n{'-'*80}")
    print(f"Column: {col}")
    print(f"{'-'*80}")
    print(f"Data Type: {df[col].dtype}")
    print(f"Unique Values: {df[col].nunique()}")
    print(f"Missing Values: {df[col].isnull().sum()}")
    
    if df[col].dtype in ['int64', 'float64']:
        print(f"\nNumerical Statistics:")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  Median: {df[col].median():.4f}")
        print(f"  Mode: {df[col].mode().values[0] if len(df[col].mode()) > 0 else 'N/A'}")
        print(f"  Std Dev: {df[col].std():.4f}")
        print(f"  Variance: {df[col].var():.4f}")
        print(f"  Min: {df[col].min():.4f}")
        print(f"  Max: {df[col].max():.4f}")
        print(f"  Range: {df[col].max() - df[col].min():.4f}")
        print(f"  25th Percentile (Q1): {df[col].quantile(0.25):.4f}")
        print(f"  50th Percentile (Q2/Median): {df[col].quantile(0.50):.4f}")
        print(f"  75th Percentile (Q3): {df[col].quantile(0.75):.4f}")
        print(f"  IQR (Q3-Q1): {df[col].quantile(0.75) - df[col].quantile(0.25):.4f}")
        print(f"  Skewness: {df[col].skew():.4f}")
        print(f"  Kurtosis: {df[col].kurtosis():.4f}")
        
        # Outlier Detection using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        print(f"  Outliers (IQR method): {len(outliers)} ({(len(outliers)/len(df))*100:.2f}%)")
        
    else:
        print(f"\nCategorical Statistics:")
        print(f"  Value Counts:\n{df[col].value_counts()}")

# ============================================================================
# 7. CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("7. CORRELATION ANALYSIS")
print("="*80)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numeric_cols].corr()
print(f"\nCorrelation Matrix:")
print(correlation_matrix)

print(f"\nCorrelation with Target Variable (Delivery_Category_Encoded):")
target_corr = correlation_matrix[target_col].sort_values(ascending=False)
print(target_corr)

print(f"\nTop 5 Features Most Correlated with Target:")
print(target_corr.drop(target_col).head(5))

# ============================================================================
# 8. FEATURE VALUE RANGES
# ============================================================================
print("\n" + "="*80)
print("8. FEATURE VALUE RANGES")
print("="*80)
for col in numeric_cols:
    print(f"{col}: [{df[col].min():.2f}, {df[col].max():.2f}]")

# ============================================================================
# 9. ZERO/CONSTANT VALUES
# ============================================================================
print("\n" + "="*80)
print("9. ZERO AND CONSTANT VALUE ANALYSIS")
print("="*80)
for col in df.columns:
    zero_count = (df[col] == 0).sum()
    zero_percent = (zero_count / len(df)) * 100
    if zero_count > 0:
        print(f"{col}: {zero_count} zeros ({zero_percent:.2f}%)")
    
    if df[col].nunique() == 1:
        print(f"⚠️ {col} has CONSTANT value: {df[col].unique()[0]}")

# ============================================================================
# 10. DATA DISTRIBUTION SUMMARY
# ============================================================================
print("\n" + "="*80)
print("10. DATA DISTRIBUTION SUMMARY")
print("="*80)
for col in numeric_cols:
    skew = df[col].skew()
    if abs(skew) < 0.5:
        dist = "Approximately Normal"
    elif skew > 0:
        dist = "Right Skewed (Positive)"
    else:
        dist = "Left Skewed (Negative)"
    print(f"{col}: {dist} (Skewness: {skew:.4f})")

# ============================================================================
# 11. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("11. RECOMMENDATIONS FOR MODEL TRAINING")
print("="*80)

# Class Imbalance
if imbalance_ratio > 1.5:
    print("\n⚠️ CLASS IMBALANCE DETECTED:")
    print(f"   - Use SMOTE or other resampling techniques")
    print(f"   - Use stratified train-test split")
    print(f"   - Consider class_weight='balanced' in models")

# Outliers
outlier_cols = []
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    if len(outliers) > len(df) * 0.05:  # More than 5% outliers
        outlier_cols.append(col)

if outlier_cols:
    print(f"\n⚠️ OUTLIERS DETECTED IN: {outlier_cols}")
    print(f"   - Consider outlier treatment or robust scalers")

# Skewed Features
skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 1]
if skewed_cols:
    print(f"\n⚠️ HIGHLY SKEWED FEATURES: {skewed_cols}")
    print(f"   - Consider log transformation or PowerTransformer")

# Scaling
print(f"\n✅ FEATURE SCALING REQUIRED:")
print(f"   - Use StandardScaler for Logistic Regression")
print(f"   - Tree-based models don't require scaling")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)