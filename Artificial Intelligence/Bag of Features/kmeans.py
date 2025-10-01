import numpy as np

class KMeans:
  def __init__(self, n_grupos, n_iter=10):
    self.n_grupos = n_grupos
    self.n_iter = n_iter

  def asigna_puntos_a_centroides(self, X):
    grupos = np.zeros(X.shape[0])
    dists = np.zeros(X.shape[0])
    for i,p in enumerate(X):
      grupos[i] = np.linalg.norm(self.centroides - p, axis = 1).argmin()
      dists[i] = np.linalg.norm(self.centroides - p, axis = 1).min()
    return grupos, dists

  def recalcula_centroides(self, X):
    for i in range(self.n_grupos):
      if(np.any(np.where(self.grupos == i))):
        self.centroides[i,:] = X[np.where(self.grupos == i), :].mean(axis = 1)


  def fit(self, X, grafica=None):
    self.centroides = np.zeros((self.n_grupos, X.shape[1]))

    ## Inicializa centroides con puntos del conjunto elegidos aleatoriamente
    permutacion = np.random.permutation(X.shape[0])
    self.centroides[:, :] = X[permutacion[:self.n_grupos], :]

    es = np.zeros(self.n_iter + 1)
    for it in range(self.n_iter):
      self.grupos, dists = self.asigna_puntos_a_centroides(X)
      es[it] = np.mean(dists**2)
      self.recalcula_centroides(X)

    self.grupos, dists = self.asigna_puntos_a_centroides(X)
    es[-1] = np.mean(dists**2)
    print(es)
    return self.centroides
