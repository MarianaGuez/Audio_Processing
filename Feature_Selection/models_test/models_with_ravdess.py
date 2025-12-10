import numpy as np
import torch

from MLP import accuracy, resultados, MLP, set_seed
from random_forest import rf_with_early_stop, k_fold_cv_rf
from SVC import k_fold_cv_svc
from sklearn.metrics import accuracy_score
import joblib
set_seed()
complete_Ravedess_train =  np.load("data/Ravdess_Train_all.npy", allow_pickle=True) 
y_Ravedess_train = np.load('data/y_Ravdess.npy', allow_pickle=True)[:238]

complete_Ravedess_test =  np.load("data/Ravdess_Test_all.npy", allow_pickle=True) 
y_Ravedess_test = np.load('data/y_Ravdess.npy', allow_pickle=True)[238:]

Ravdess_all = torch.tensor(complete_Ravedess_train, dtype=torch.float32)
y_Ravedess_train = torch.tensor(y_Ravedess_train, dtype=torch.long)

Ravedess_test = torch.tensor(complete_Ravedess_test, dtype=torch.float32)
y_Ravedess_test = torch.tensor(y_Ravedess_test, dtype=torch.long)

rav_all_to_all_features = np.load('data/rav_all_to_all_features.npy', allow_pickle=True).item()
mask_Ravadess = np.array(list(rav_all_to_all_features.values())).astype(bool)

Ravedess_selected = complete_Ravedess_train[:,mask_Ravadess]
Ravadess_selected_test = Ravedess_test[:,mask_Ravadess]
Ravedess_selected = torch.tensor(Ravedess_selected, dtype=torch.float32)

#MLP All features

resultados(Ravdess_all, y_Ravedess_train, k=5, epochs=300, lr=0.01, early_stop=50, fold="All")


model = MLP(input_dim=411)
model.load_state_dict(torch.load("data/best_mlp_model.pth", map_location="cpu"))
print("MLP Test All features:", accuracy(model, Ravedess_test, y_Ravedess_test))

#MLP Selected features

resultados(Ravedess_selected, y_Ravedess_train, k=5, epochs=300, lr=0.01, early_stop=50, fold="Selected")

model = MLP(input_dim=Ravadess_selected_test.shape[1])
model.load_state_dict(torch.load("data/best_mlp_model.pth", map_location="cpu"))
print("MLP Test Selected features:", accuracy(model, Ravadess_selected_test, y_Ravedess_test))
print()

#Random forest All features

fold_scores, best_model = k_fold_cv_rf(Ravdess_all, y_Ravedess_train, k=5, max_estimators=200, step=10, patience=3, random_state=42) 
print("Random forest Valid All features:", np.mean(fold_scores))

best_rf_model = joblib.load("data/best_rf_model.pkl")
y_pred = best_rf_model.predict(Ravedess_test)

acc = accuracy_score(y_Ravedess_test, y_pred)
print(f"Random forest Test All features: {acc:.4f}\n")


#Random forest Selected features

fold_scores, best_model = k_fold_cv_rf(Ravedess_selected, y_Ravedess_train, k=5, max_estimators=200, step=10, patience=3, random_state=42) 
print("Random forest Valid Selected features:", np.mean(fold_scores))

best_rf_model = joblib.load("data/best_rf_model.pkl")
y_pred = best_rf_model.predict(Ravadess_selected_test)
acc = accuracy_score(y_Ravedess_test, y_pred)
print(f"Random forest Test Selected features: {acc:.4f}")


#SVC All features

scores, best_model = k_fold_cv_svc(Ravdess_all, y_Ravedess_train, k=5)
print(f"SVC Valid All features: {np.mean(scores):.4f}")
best_rf_model = joblib.load("data/best_svc_model.pkl")

y_pred = best_rf_model.predict(Ravedess_test)
acc = accuracy_score(y_Ravedess_test, y_pred)
print(f"SVC Test All features: {acc:.4f}\n")


#SVC selected features

scores, best_model = k_fold_cv_svc(Ravedess_selected, y_Ravedess_train, k=5)
print(f"SVC Valid Selected features: {np.mean(scores):.4f}")
best_rf_model = joblib.load("data/best_svc_model.pkl")
y_pred = best_rf_model.predict(Ravadess_selected_test)
acc = accuracy_score(y_Ravedess_test, y_pred)
print(f"SVC Test Selected features: {acc:.4f}")






