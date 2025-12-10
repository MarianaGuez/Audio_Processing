import numpy as np
import torch

from MLP import accuracy, resultados, MLP, set_seed
from random_forest import rf_with_early_stop, k_fold_cv_rf
from SVC import k_fold_cv_svc
from sklearn.metrics import accuracy_score
import joblib
set_seed()
complete_CreamD_train =  np.load("data/CreamD_Train_all.npy", allow_pickle=True) 
y_CreamD_train = np.load('data/y_CreamD.npy', allow_pickle=True)[:238]

complete_CreamD_test =  np.load("data/CreamD_Test_all.npy", allow_pickle=True) 
y_CreamD_test = np.load('data/y_CreamD.npy', allow_pickle=True)[238:]

CreamD_all = torch.tensor(complete_CreamD_train, dtype=torch.float32)
y_CreamD_train = torch.tensor(y_CreamD_train, dtype=torch.long)

CreamD_test = torch.tensor(complete_CreamD_test, dtype=torch.float32)
y_CreamD_test = torch.tensor(y_CreamD_test, dtype=torch.long)

rav_all_to_all_features = np.load('data/crema_all_to_all_features.npy', allow_pickle=True).item()
mask_Ravadess = np.array(list(rav_all_to_all_features.values())).astype(bool)

CreamD_selected = complete_CreamD_train[:,mask_Ravadess]
Ravadess_selected_test = CreamD_test[:,mask_Ravadess]
CreamD_selected = torch.tensor(CreamD_selected, dtype=torch.float32)

#MLP All features

resultados(CreamD_all, y_CreamD_train, k=5, epochs=300, lr=0.01, early_stop=50, fold="All")


model = MLP(input_dim=411)
model.load_state_dict(torch.load("data/best_mlp_model.pth", map_location="cpu"))
print("MLP Test All features:", accuracy(model, CreamD_test, y_CreamD_test))

#MLP Selected features

resultados(CreamD_selected, y_CreamD_train, k=5, epochs=300, lr=0.01, early_stop=50, fold="Selected")

model = MLP(input_dim=Ravadess_selected_test.shape[1])
model.load_state_dict(torch.load("data/best_mlp_model.pth", map_location="cpu"))
print("MLP Test Selected features:", accuracy(model, Ravadess_selected_test, y_CreamD_test))
print()

#Random forest All features

fold_scores, best_model = k_fold_cv_rf(CreamD_all, y_CreamD_train, k=5, max_estimators=200, step=10, patience=3, random_state=42) 
print("Random forest Valid All features:", np.mean(fold_scores))

best_rf_model = joblib.load("data/best_rf_model.pkl")
y_pred = best_rf_model.predict(CreamD_test)

acc = accuracy_score(y_CreamD_test, y_pred)
print(f"Random forest Test All features: {acc:.4f}\n")


#Random forest Selected features

fold_scores, best_model = k_fold_cv_rf(CreamD_selected, y_CreamD_train, k=5, max_estimators=200, step=10, patience=3, random_state=42) 
print("Random forest Valid Selected features:", np.mean(fold_scores))

best_rf_model = joblib.load("data/best_rf_model.pkl")
y_pred = best_rf_model.predict(Ravadess_selected_test)
acc = accuracy_score(y_CreamD_test, y_pred)
print(f"Random forest Test Selected features: {acc:.4f}")


#SVC All features

scores, best_model = k_fold_cv_svc(CreamD_all, y_CreamD_train, k=5)
print(f"SVC Valid All features: {np.mean(scores):.4f}")
best_rf_model = joblib.load("data/best_svc_model.pkl")

y_pred = best_rf_model.predict(CreamD_test)
acc = accuracy_score(y_CreamD_test, y_pred)
print(f"SVC Test All features: {acc:.4f}\n")


#SVC selected features

scores, best_model = k_fold_cv_svc(CreamD_selected, y_CreamD_train, k=5)
print(f"SVC Valid Selected features: {np.mean(scores):.4f}")
best_rf_model = joblib.load("data/best_svc_model.pkl")
y_pred = best_rf_model.predict(Ravadess_selected_test)
acc = accuracy_score(y_CreamD_test, y_pred)
print(f"SVC Test Selected features: {acc:.4f}")
