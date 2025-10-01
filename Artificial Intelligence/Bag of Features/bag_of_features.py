from feature_extraction import extract_features
from functions import array_normalization, histogram
from kmeans import KMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import chi2_kernel

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


directorio_Test = 'Proyecto_data/Test'
directorio_Train = 'Proyecto_data/Train'

dataTr, y_train = extract_features(directorio_Train)
dataTe, y_test = extract_features(directorio_Test)

Train_set = array_normalization(dataTr)
Test_set = array_normalization(dataTe)

mbkm_places = KMeans(n_grupos=7000)
centroides = mbkm_places.fit(Train_set)

histTr = histogram(y_train,Train_set, centroides)
histTe = histogram(y_test,Test_set, centroides)

X_train = array_normalization(histTr)
X_test = array_normalization(histTe)

clf = svm.SVC(kernel = chi2_kernel, C=4)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

corr = confusion_matrix(y_test, y_pred)

#df = pd.DataFrame(corr)
emotions = ["Hopeful", "Excited", "Happy", "Sad"]

sb.heatmap(
    corr, 
    cmap="Blues", 
    annot=True, 
    xticklabels=emotions, 
    yticklabels=emotions
)

plt.show()

#sb.heatmap(corr, cmap="Blues", annot=True, xticklabels='SEFT', yticklabels='SEFT')

#plt.show()
