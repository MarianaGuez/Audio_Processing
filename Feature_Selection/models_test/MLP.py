import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random


class MLP(nn.Module):
    def __init__(self, input_dim=411, hidden_dim=100, hidden_dim_two = 50, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim_two)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim_two, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
        
def train_fold(model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs=50, early_stop=100):
    train_losses, val_losses = [], []
    EARLY_STOP = 0
    best_val = float('inf')
    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

       
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            val_losses.append(val_loss.item())
           
        if val_loss < best_val:
            best_val = val_loss
            EARLY_STOP = 0
        else:
            EARLY_STOP += 1
            if EARLY_STOP == early_stop:
                print(f"Early stopping at epoch {epoch+1}")
                torch.save(model.state_dict(), 'data/best_mlp_model.pth')
                break

    return train_losses, val_losses

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        preds = outputs.argmax(dim=1)
    return (preds == y).float().mean().item()

def set_seed(seed=42):
    random.seed(seed)             
    np.random.seed(seed)           
    torch.manual_seed(seed)       

def k_fold_cv(X, y, k=5, epochs=50, lr=0.01, early_stop=100):
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // k

    all_train_losses = []
    all_val_losses = []
    scores = []

    for fold in range(k):
        print(f"Fold {fold+1}/{k}")
        val_idx = indices[fold*fold_size:(fold+1)*fold_size]
        train_idx = np.setdiff1d(indices, val_idx)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = MLP(input_dim=X_train.shape[1])
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses = train_fold(
            model, optimizer, criterion, X_train, y_train, X_val, y_val, epochs, early_stop=early_stop
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        acc = accuracy(model, X_val, y_val)
        scores.append(acc)

    return all_train_losses, all_val_losses, scores


def graficar_loss(train_losses, val_losses,k):

    plt.figure(figsize=(10,6))
    for i in range(k):
        plt.plot(train_losses[i], label=f'Train Fold {i+1}')
        plt.plot(val_losses[i], linestyle='--', label=f'Val Fold {i+1}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training - Validation Loss por Fold")
    plt.legend()
    plt.show()

def resultados(X,y, k=5, epochs=50, lr=0.01, early_stop=100, fold="all"):
  set_seed(42) 
  train_losses, val_losses, scores = k_fold_cv(X, y, k=k, epochs=epochs, lr=lr, early_stop=early_stop)

  print("Scores en cada fold:", scores)
  print(f"MLP Valid {fold}: {np.mean(scores)}")
  graficar_loss(train_losses, val_losses,k)
  
  

