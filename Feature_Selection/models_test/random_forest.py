from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import copy
import numpy as np
import joblib
from MLP import set_seed
set_seed()
def rf_with_early_stop(X_train, y_train, X_val, y_val, 
                       max_estimators=200, step=10, patience=3):
    best_acc = 0
    patience_counter = 0
    scores = []
    best_model = None
    
    model = RandomForestClassifier(
        n_estimators=0,
        warm_start=True,
        random_state=42
    )

    for n in range(step, max_estimators+1, step):
        model.n_estimators = n
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            best_model = copy.deepcopy(model)  # clonar modelo actual
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    
    return best_model, scores


def k_fold_cv_rf(X, y, k=5, max_estimators=200, step=10, patience=3, random_state=42):
    set_seed(42) 
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k

    fold_scores = []
    best_global_score = float('-inf')
    best_global_model = None

    for fold in range(k):
        print(f"Fold {fold+1}/{k}")
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        best_model_fold, scores_fold = rf_with_early_stop(
            X_train, y_train, X_val, y_val,
            max_estimators=max_estimators,
            step=step,
            patience=patience
        )

        acc = max(scores_fold)
        fold_scores.append(acc)

      
        if acc > best_global_score:
            best_global_score = acc
            best_global_model = best_model_fold


    joblib.dump(best_global_model, "data/best_rf_model.pkl")
    print("Saved best model as 'best_rf_model.pkl'")

    return fold_scores, best_global_model

