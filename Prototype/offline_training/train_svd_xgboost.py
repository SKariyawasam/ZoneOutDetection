import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
import pickle

def main():
    print("Loading extracted SVD features...")
    df = pd.read_csv("svd_features_train.csv")
    
    # Separate features and labels
    y = df['label']
    # Drop non-feature columns
    X = df.drop(columns=['label', 'video_path'])
    
    print(f"Dataset shape: {X.shape}")
    
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.model_selection import StratifiedKFold
    import numpy as np
    
    # 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_bal_accs, cv_f1s, cv_aucs, cv_praucs = [], [], [], [], []
    
    print("\n--- Running 5-Fold Stratified Cross-Validation on Balanced SVD Dataset ---")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_val_f = X.iloc[val_idx], y.iloc[val_idx]
        
        sw_tr = compute_sample_weight('balanced', y_tr)
        
        m = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            eval_metric='logloss',
            early_stopping_rounds=30,
            random_state=42
        )
        
        m.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_va, y_val_f)],
            verbose=False
        )
        
        preds_f = m.predict(X_va)
        probs_f = m.predict_proba(X_va)[:, 1]
        
        from sklearn.metrics import balanced_accuracy_score, average_precision_score
        f_acc = accuracy_score(y_val_f, preds_f)
        f_bal = balanced_accuracy_score(y_val_f, preds_f)
        f_f1 = f1_score(y_val_f, preds_f, average='macro')
        f_auc = roc_auc_score(y_val_f, probs_f)
        f_pr = average_precision_score(y_val_f, probs_f)
        
        cv_accs.append(f_acc)
        cv_bal_accs.append(f_bal)
        cv_f1s.append(f_f1)
        cv_aucs.append(f_auc)
        cv_praucs.append(f_pr)
        print(f"  Fold {fold+1}: Balanced Acc={f_bal:.4f}, Macro F1={f_f1:.4f}, ROC AUC={f_auc:.4f}, PR AUC={f_pr:.4f}")
        
    print("\n=======================================================")
    print("5-Fold Cross-Validation Summary (Mean ± Std):")
    print(f"  Raw Accuracy     : {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")
    print(f"  Balanced Accuracy: {np.mean(cv_bal_accs):.4f} ± {np.std(cv_bal_accs):.4f}")
    print(f"  Macro F1 Score   : {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}")
    print(f"  ROC AUC          : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"  PR AUC           : {np.mean(cv_praucs):.4f} ± {np.std(cv_praucs):.4f}")
    print("=======================================================\n")
    
    # Train final deployment model on full dataset
    print("Training final XGBoost model on full dataset with balanced class weighting...")
    sw_all = compute_sample_weight('balanced', y)
    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X, y, sample_weight=sw_all)
    
    # Save the model
    model_path = "svd_xgboost_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")
    
    # Save the feature columns list so inference knows exactly what to pass
    cols_path = "svd_feature_cols.pkl"
    with open(cols_path, 'wb') as f:
        pickle.dump(list(X.columns), f)
    print(f"Feature columns saved to {cols_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during training: {e}")
