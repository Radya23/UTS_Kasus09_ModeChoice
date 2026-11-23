import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from fitur import reshape_long_to_wide, buat_fitur_baru, get_raw_feature_columns

DATA_PATH = os.path.join("dataset", "kasus09_mode_choice.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_mode_choice.joblib")

def main():
    # 1. Load & Reshape Long -> Wide
    df_long = pd.read_csv(DATA_PATH)
    wide_df = reshape_long_to_wide(df_long)
    
    # Buang kolom ID ('individual') dan pisahkan target ('target_mode')
    raw_cols = get_raw_feature_columns()
    X_raw = wide_df[raw_cols]
    y = wide_df['target_mode']
    
    # 2. Train-Test Split SEBELUM preprocessing (imputasi/scaling)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # 3. Terapkan Feature Engineering
    X_train = buat_fitur_baru(X_train_raw)
    X_test = buat_fitur_baru(X_test_raw)
    
    # 4. Definisikan Pipeline Model untuk Perbandingan
    pipelines = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
        ])
    }
    
    # 5. Pemilihan Model dengan Stratified Cross-Validation di Data Train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    
    print("=== HASIL CROSS-VALIDATION (DATA TRAIN) ===")
    for name, pipe in pipelines.items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1_macro"]
        )
        mean_acc = float(np.mean(scores["test_accuracy"]))
        std_acc = float(np.std(scores["test_accuracy"]))
        mean_f1 = float(np.mean(scores["test_f1_macro"]))
        std_f1 = float(np.std(scores["test_f1_macro"]))
        
        cv_results[name] = {
            "cv_accuracy_mean": mean_acc,
            "cv_accuracy_std": std_acc,
            "cv_f1_macro_mean": mean_f1,
            "cv_f1_macro_std": std_f1
        }
        print(f"{name:20s} | Accuracy: {mean_acc:.4f} (±{std_acc:.4f}) | Macro F1: {mean_f1:.4f} (±{std_f1:.4f})")
        
    # Pilih model terbaik berdasarkan Macro F1 CV (agar adil untuk kelas minoritas Moda 3)
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["cv_f1_macro_mean"])
    best_pipeline = pipelines[best_model_name]
    print(f"\nModel Terpilih: {best_model_name}")
    
    # 6. Latih Model Terpilih pada Seluruh Data Train & Evaluasi SEKALI pada Data Test
    best_pipeline.fit(X_train, y_train)
    y_pred_test = best_pipeline.predict(X_test)
    
    test_acc = float(accuracy_score(y_test, y_pred_test))
    test_f1_macro = float(f1_score(y_test, y_pred_test, average="macro"))
    test_f1_weighted = float(f1_score(y_test, y_pred_test, average="weighted"))
    cm = confusion_matrix(y_test, y_pred_test, labels=[1, 2, 3, 4]).tolist()
    report_dict = classification_report(y_test, y_pred_test, output_dict=True)
    
    print("\n=== EVALUASI FINAL PADA DATA TEST (1x) ===")
    print(f"Test Accuracy    : {test_acc:.4f}")
    print(f"Test Macro F1    : {test_f1_macro:.4f}")
    print(f"Test Weighted F1 : {test_f1_weighted:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))
    
    # 7. Simpan Pipeline + Metadata dengan Joblib
    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "raw_features": raw_cols,
        "engineered_features": list(X_train.columns),
        "classes": [1, 2, 3, 4],
        "class_distribution": y.value_counts().sort_index().to_dict(),
        "cv_results": cv_results,
        "test_metrics": {
            "accuracy": test_acc,
            "f1_macro": test_f1_macro,
            "f1_weighted": test_f1_weighted,
            "confusion_matrix": cm,
            "classification_report": report_dict
        },
        "default_sample": X_test_raw.iloc[0].to_dict()
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nArtifact berhasil disimpan ke: {MODEL_PATH}")

if __name__ == "__main__":
    main()
