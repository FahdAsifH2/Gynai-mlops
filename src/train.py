"""
GynAI Delivery Mode Prediction - Optimized Pipeline
Experiment 1: Stacking Ensemble with Advanced Preprocessing
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, classification_report, 
                            confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import mlflow
import mlflow.sklearn

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'data_path': 'maternal_data_cleaned_encoded.csv',
    'target_col': 'Delivery_Category_Encoded',
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'imbalance_ratio': 2.23
}

# ============================================================================
# 1. DATA LOADING & CLEANING
# ============================================================================
def load_and_clean_data():
    print("\n" + "="*70)
    print("📊 LOADING DATA")
    print("="*70)
    
    df = pd.read_csv(CONFIG['data_path'])
    
    # Remove duplicates
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    removed = initial_shape - df.shape[0]
    
    print(f"✓ Loaded: {initial_shape} rows, {df.shape[1]} columns")
    print(f"✓ Removed duplicates: {removed} rows")
    print(f"✓ Final dataset: {df.shape[0]} rows")
    
    return df

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
def engineer_features(df):
    print("\n" + "="*70)
    print("🔧 FEATURE ENGINEERING")
    print("="*70)
    
    df = df.copy()
    
    # Age Risk Groups
    df['Age_Risk'] = pd.cut(df['    AgeAtStartOfSpell'], 
                            bins=[0, 20, 35, 100], labels=[0, 1, 2]).astype(int)
    
    # BMI Categories
    df['BMI_Category'] = pd.cut(df['Body Mass Index at Booking'],
                                bins=[0, 18.5, 25, 30, 100], 
                                labels=[0, 1, 2, 3]).astype(int)
    
    # Previous Delivery Risk
    df['Prev_Delivery_Risk'] = (df['No_Of_previous_Csections'] + 
                                (df['Parity'] > 3).astype(int))
    
    # High Risk Score
    df['High_Risk_Score'] = ((df['Obese_Encoded'] * 2) +
                            (df['GestationalDiabetes_Encoded'] * 2) +
                            (df['No_Of_previous_Csections'] > 0).astype(int) +
                            (df['    AgeAtStartOfSpell'] > 35).astype(int))
    
    # Gestation Risk
    df['Gestation_Risk'] = pd.cut(df['Gestation (Days)'],
                                  bins=[0, 259, 280, 500],
                                  labels=[0, 1, 2]).astype(int)
    
    # Weight to Height Ratio
    df['Weight_Height_Ratio'] = df['WeightMeasured'] / df['Height']
    
    print(f"✓ Created 6 new features")
    print(f"✓ Total features: {df.shape[1]}")
    
    return df

# ============================================================================
# 3. PREPROCESSING PIPELINE
# ============================================================================
def preprocess_data(df):
    print("\n" + "="*70)
    print("⚙️  PREPROCESSING")
    print("="*70)
    
    # Separate target
    X = df.drop(CONFIG['target_col'], axis=1)
    y = df[CONFIG['target_col']]
    
    # Class distribution
    class_dist = y.value_counts()
    print(f"✓ Class 0 (Normal): {class_dist[0]} ({class_dist[0]/len(y)*100:.1f}%)")
    print(f"✓ Class 1 (C-Section): {class_dist[1]} ({class_dist[1]/len(y)*100:.1f}%)")
    
    # Handle skewness
    skewed_cols = ['WeightMeasured', 'Body Mass Index at Booking', 
                   'Parity', 'Gravida', 'No_Of_previous_Csections']
    
    pt = PowerTransformer(method='yeo-johnson')
    X[skewed_cols] = pt.fit_transform(X[skewed_cols])
    print(f"✓ Applied PowerTransformer to {len(skewed_cols)} skewed features")
    
    # Robust scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    print(f"✓ Applied RobustScaler to all features")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    
    # SMOTE for class imbalance
    smote = SMOTE(random_state=CONFIG['random_state'])
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    added = len(X_train_balanced) - len(X_train)
    print(f"✓ SMOTE applied: +{added} synthetic samples")
    print(f"✓ Balanced train set: {X_train_balanced.shape[0]} samples")
    
    return X_train_balanced, X_test, y_train_balanced, y_test, scaler, pt

# ============================================================================
# 4. MODEL TRAINING
# ============================================================================
def train_models(X_train, y_train):
    print("\n" + "="*70)
    print("🤖 TRAINING MODELS")
    print("="*70)
    
    # Base Models
    xgb = XGBClassifier(
        learning_rate=0.05,
        max_depth=7,
        n_estimators=300,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=CONFIG['imbalance_ratio'],
        random_state=CONFIG['random_state'],
        eval_metric='logloss',
        verbosity=0
    )
    
    lgbm = LGBMClassifier(
        learning_rate=0.05,
        num_leaves=50,
        max_depth=10,
        n_estimators=300,
        min_data_in_leaf=20,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        is_unbalance=True,
        random_state=CONFIG['random_state'],
        verbose=-1
    )
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=CONFIG['random_state'],
        n_jobs=-1
    )
    
    # Stacking Ensemble
    print("Training Stacking Ensemble...")
    print("  ├─ Base: XGBoost")
    print("  ├─ Base: LightGBM")
    print("  ├─ Base: Random Forest")
    print("  └─ Meta: Logistic Regression")
    
    stacking_clf = StackingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('rf', rf)
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=CONFIG['random_state']),
        cv=CONFIG['cv_folds'],
        n_jobs=-1
    )
    
    stacking_clf.fit(X_train, y_train)
    print("✓ Stacking Ensemble trained successfully")
    
    return stacking_clf

# ============================================================================
# 5. MODEL EVALUATION
# ============================================================================
def evaluate_model(model, X_test, y_test):
    print("\n" + "="*70)
    print("📈 MODEL EVALUATION")
    print("="*70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Print results
    print(f"\n{'Metric':<20} {'Score':<10} {'Status'}")
    print("-" * 50)
    print(f"{'Accuracy':<20} {accuracy:>6.2%}    {'✅' if accuracy >= 0.90 else '⚠️'}")
    print(f"{'Precision':<20} {precision:>6.2%}    {'✅' if precision >= 0.85 else '⚠️'}")
    print(f"{'Recall (Critical)':<20} {recall:>6.2%}    {'✅' if recall >= 0.85 else '⚠️'}")
    print(f"{'F1-Score':<20} {f1:>6.2%}    {'✅' if f1 >= 0.90 else '⚠️'}")
    print(f"{'ROC-AUC':<20} {roc_auc:>6.2%}    {'✅' if roc_auc >= 0.90 else '⚠️'}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Normal  C-Section")
    print(f"Actual Normal    {cm[0,0]:<6}  {cm[0,1]:<6}")
    print(f"       C-Section {cm[1,0]:<6}  {cm[1,1]:<6}")
    
    # Class-wise metrics
    print(f"\nClass-wise Performance:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'C-Section'],
                                digits=4))
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    
    return metrics

# ============================================================================
# 6. MLFLOW LOGGING
# ============================================================================
def log_to_mlflow(model, metrics, X_train):
    print("\n" + "="*70)
    print("📊 LOGGING TO MLFLOW")
    print("="*70)
    
    mlflow.set_experiment("experiment_1")
    
    with mlflow.start_run(run_name="stacking_ensemble_optimal"):
        
        # Log parameters
        mlflow.log_param("model_type", "Stacking_Ensemble")
        mlflow.log_param("base_models", "XGBoost_LightGBM_RandomForest")
        mlflow.log_param("meta_model", "LogisticRegression")
        mlflow.log_param("preprocessing", "PowerTransformer_RobustScaler")
        mlflow.log_param("resampling", "SMOTE")
        mlflow.log_param("cv_folds", CONFIG['cv_folds'])
        mlflow.log_param("test_size", CONFIG['test_size'])
        mlflow.log_param("features_engineered", 6)
        mlflow.log_param("total_features", X_train.shape[1])
        
        # Log metrics
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("roc_auc", metrics['roc_auc'])
        
        # Log model
        mlflow.sklearn.log_model(model, "stacking_model")
        
        print("✓ Parameters logged")
        print("✓ Metrics logged")
        print("✓ Model logged")
        print(f"✓ Run ID: {mlflow.active_run().info.run_id}")

# ============================================================================
# 7. MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "="*70)
    print(" 🏥 GYNAI DELIVERY MODE PREDICTION - EXPERIMENT 1")
    print("="*70)
    print(" Architecture: Stacking Ensemble + Advanced Preprocessing")
    print(" Target: 90%+ Accuracy & 85%+ Recall")
    print("="*70)
    
    # Pipeline
    df = load_and_clean_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler, pt = preprocess_data(df)
    model = train_models(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    log_to_mlflow(model, metrics, X_train)
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    print(f"View results: mlflow ui")
    print(f"Then open: http://localhost:5000")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()