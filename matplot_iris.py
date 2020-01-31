# -*- coding: utf-8 -*-
"""Untitled

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f3ewu9kbT-VZ5zgy8hjMFmI-0RYTAHho
"""



"""**Train Test Split**"""

import warnings

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=DeprecationWarning)

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

# logistic_reg
lgr = LogisticRegression(solver='liblinear', multi_class='auto')
lgr.fit(X_train, y_train)
y_pred = lgr.predict(X_test)
score_lr = accuracy_score(y_test, y_pred)

# KNN
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 50)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X_train, y_train)
best_n = knn_gscv.best_params_['n_neighbors']


knn = KNeighborsClassifier(n_neighbors=best_n)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
score_knn = accuracy_score(y_test, y_pred_knn)


print("logistic_regression:", score_lr)
print("KNN:", score_knn)

"""**KNN (n = 1 to 30) -> can you find, which ‘n’ provides the highest accuracy?**"""

current_score=0
high_score=0;
high_n=0
scores=[]

for i in range (1,31):
  knn = KNeighborsClassifier(n_neighbors=i)
  knn.fit(X_train, y_train)
  y_pred_knn = knn.predict(X_test)
  score_knn = accuracy_score(y_test, y_pred_knn)
  scores.append(score_knn)
  if score_knn>current_score:
    high_score=score_knn
    high_n=i
    current_score=high_score

  print("n:",i,"Accuracy:",score_knn)
print("Highest score:",high_score,"while n:",high_n)

"""**Can you plot a simple graph, where K values (1-30) form your x-axis and accuracy score 0-1 form your y-axis?**"""

import matplotlib.pyplot as plt
plt.plot(np.arange(1,31),scores,color='orange', marker='o', linestyle='dashed',linewidth=2, markersize=12)
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")

plt.axis([7, 15, 0, 1.5])
plt.show()


# K-fold CV

from sklearn.model_selection import cross_val_score
score_knnX=[]
for i in range(1,45):

  knnX=KNeighborsClassifier(n_neighbors=i)
  cross_score_knn=cross_val_score(knnX, X, y, cv=10, scoring='accuracy')
  score_knnX.append(cross_score_knn.mean())
  print(i,cross_score_knn.mean())



plt.plot(np.arange(1,45),score_knnX)
plt.xlabel("n_neighbors")
plt.ylabel("cross_accuracy")

plt.show()



from sklearn.model_selection import KFold
kf=KFold(n_splits=3,shuffle=True)
kf.get_n_splits(X)
print(kf)

for train_index, test_index in kf.split(X):
  print("TRAIN:", train_index ,"TEST:", test_index)
  X_trainn, X_testt = X[train_index], X[test_index]
  y_trainn, y_testt= y[train_index], y[test_index]



print(iris.keys())
print(iris.feature_names)
print(iris.target_names)
print(iris.filename)
print(iris.data.shape)
print(iris.target.shape)

n_samples, n_features = iris.data.shape

print(type(iris.data))

import seaborn as sns
iris2=sns.load_dataset('iris')
correlation=iris2.corr()
print(correlation)


