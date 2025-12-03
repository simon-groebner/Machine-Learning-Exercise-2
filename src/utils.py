import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

runtimes= {}

def add_runtime(label, duration):
    # overwrites previous value for the same label
    runtimes[label] = duration

def print_runtimes():
    print("\n--- Runtime summary ---")
    for label, dur in runtimes.items():
        print(f"{label}: {dur:.4f} s")

def evaluate_model(model, X_train, y_train, X_test, y_test, n_splits=5, seed=123):
    """
    1. Runs K-Fold CV on X_train.
    2. Trains model on full X_train.
    3. Tests model on X_test.
    4. Returns a dictionary with CV scores AND Test results for plotting.
    """
    
    # CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_r2_scores = []
    cv_mse_scores = []
    
    for train_idx, val_idx in kf.split(X_train):
        # Slice the Training Data further for CV
        X_fold_t, y_fold_t = X_train[train_idx], y_train[train_idx]
        X_fold_v, y_fold_v = X_train[val_idx], y_train[val_idx]
        
        model.fit(X_fold_t, y_fold_t)
        fold_preds = model.predict(X_fold_v)
        
        cv_r2_scores.append(r2_score(y_fold_v, fold_preds))
        cv_mse_scores.append(mean_squared_error(y_fold_v, fold_preds))

    avg_cv_r2 = np.mean(cv_r2_scores)



    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)
    


    return {
        "CV_R2": avg_cv_r2,
        "CV_MSE": np.mean(cv_mse_scores),
        "Test_MSE": test_mse,
        "Test_RMSE": test_rmse,
        "Test_R2": test_r2,
        "y_test_actual": y_test,       
        "y_test_pred": y_test_pred    
    }

 
def print_results(model_name, results):
    """
    Helper function to print results
    """
    print(f"--- {model_name} ---")
    print(f"  CV MSE (Error):    {results['CV_MSE']:.4f}")
    print(f"  CV R² (Stability):    {results['CV_R2']:.4f}")
    print(f"  Test MSE (Error):     {results['Test_MSE']:.4f}")
    print(f"  Test RMSE (Error):    {results['Test_RMSE']:.4f}")
    print(f"  Test R² (Accuracy):   {results['Test_R2']:.4f}")
    print("-" * 30)