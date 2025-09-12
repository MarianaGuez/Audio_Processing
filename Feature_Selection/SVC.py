import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

from MLP import set_seed

set_seed()
   
def k_fold_cv_svc(X, y, k=5, kernel='rbf', C=10, random_state=42, save_best=True):
    SEED = random_state
    np.random.seed(SEED)

    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k

    scores = []
    best_score = -1
    best_model = None

    for fold in range(k):
        print(f"Fold {fold+1}/{k}")
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = SVC(kernel=kernel, C=C, probability=True, random_state=SEED)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

        print(f"Fold {fold+1} Accuracy: {acc:.4f}")


        if save_best and acc > best_score:
            best_score = acc
            best_model = model
            joblib.dump(best_model, "data/best_svc_model.pkl")

    
    return scores, best_model
