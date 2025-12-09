import os
import joblib
import pandas as pd
import numpy as np
import warnings
import json 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
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
CONFIG = {
    'data_path': 'data/processed.csv',
    'target_col': 'Delivery_Category_Encoded',
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'imbalance_ratio': 2.23
}

def load_and_clean_data():
    print("Loading data...")
    df = pd.read_csv(CONFIG['data_path'])
    df = df.drop_duplicates()
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def engineer_features(df):
    print("Engineering features...")
    df = df.copy()
    df['Age_Risk'] = pd.cut(df['    AgeAtStartOfSpell'], bins=[0, 20, 35, 100], labels=[0, 1, 2]).astype(int)
    df['BMI_Category'] = pd.cut(df['Body Mass Index at Booking'], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
    df['Prev_Delivery_Risk'] = (df['No_Of_previous_Csections'] + (df['Parity'] > 3).astype(int))
    df['High_Risk_Score'] = ((df['Obese_Encoded'] * 2) + (df['GestationalDiabetes_Encoded'] * 2) + 
                            (df['No_Of_previous_Csections'] > 0).astype(int) + (df['    AgeAtStartOfSpell'] > 35).astype(int))
    df['Gestation_Risk'] = pd.cut(df['Gestation (Days)'], bins=[0, 259, 280, 500], labels=[0, 1, 2]).astype(int)
    df['Weight_Height_Ratio'] = df['WeightMeasured'] / df['Height']
    print(f"Total features: {df.shape[1]}")
    return df

def preprocess_data(df):
    print("Preprocessing...")
    X = df.drop(CONFIG['target_col'], axis=1)
    y = df[CONFIG['target_col']]
    
    skewed_cols = ['WeightMeasured', 'Body Mass Index at Booking', 'Parity', 'Gravida', 'No_Of_previous_Csections']
    pt = PowerTransformer(method='yeo-johnson')
    X[skewed_cols] = pt.fit_transform(X[skewed_cols])
    
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=CONFIG['test_size'], 
                                                        random_state=CONFIG['random_state'], stratify=y)
    
    smote = SMOTE(random_state=CONFIG['random_state'])
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"Train: {X_train_balanced.shape[0]}, Test: {X_test.shape[0]}")
    return X_train_balanced, X_test, y_train_balanced, y_test, scaler, pt

def train_models(X_train, y_train):
    print("Training model...")
    
    xgb = XGBClassifier(learning_rate=0.05, max_depth=7, n_estimators=300, min_child_weight=3,
                       gamma=0.1, subsample=0.8, colsample_bytree=0.8, 
                       scale_pos_weight=CONFIG['imbalance_ratio'], random_state=CONFIG['random_state'],
                       eval_metric='logloss', verbosity=0)
    
    lgbm = LGBMClassifier(learning_rate=0.05, num_leaves=50, max_depth=10, n_estimators=300,
                         min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
                         is_unbalance=True, random_state=CONFIG['random_state'], verbose=-1)
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                               min_samples_leaf=5, class_weight='balanced',
                               random_state=CONFIG['random_state'], n_jobs=-1)
    
    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=CONFIG['random_state']),
        cv=CONFIG['cv_folds'], n_jobs=-1
    )
    
    stacking_clf.fit(X_train, y_train)
    print("Model trained")
    return stacking_clf

def evaluate_model(model, X_test, y_test):
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }
    return metrics

def log_to_mlflow(model, metrics, X_train):
    print("Logging to MLflow...")
    mlflow.set_experiment("gynai_experiment")
    
    with mlflow.start_run(run_name="stacking_ensemble"):
        mlflow.log_param("model_type", "Stacking_Ensemble")
        mlflow.log_param("total_features", X_train.shape[1])
        mlflow.log_metric("accuracy", metrics['accuracy'])
        mlflow.log_metric("precision", metrics['precision'])
        mlflow.log_metric("recall", metrics['recall'])
        mlflow.log_metric("f1_score", metrics['f1_score'])
        mlflow.log_metric("roc_auc", metrics['roc_auc'])
        mlflow.sklearn.log_model(model, "model")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
def save_metrics_to_file(metrics):
    """Save metrics to JSON file for DVC tracking"""
    os.makedirs("metrics", exist_ok=True)
    metrics_file = "metrics/metrics.json"
    
    # Convert numpy types to Python native types for JSON serialization
    metrics_dict = {k: float(v) if isinstance(v, np.generic) else v 
                   for k, v in metrics.items()}
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Metrics saved to: {metrics_file}")
def main():
    print("="*70)
    print("MAIN FUNCTION STARTED")
    print("GYNAI DELIVERY MODE PREDICTION")
    print("="*70)
    
    df = load_and_clean_data()
    df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler, pt = preprocess_data(df)
    model = train_models(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics for DVC tracking
    save_metrics_to_file(metrics)
    
    log_to_mlflow(model, metrics, X_train)
    
    print("Saving model...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    print("Model saved to: models/model.pkl")
    
    print("="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
if __name__ == "__main__":
    main()