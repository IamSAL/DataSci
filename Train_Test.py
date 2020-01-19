import warnings


import numpy as np
import tensorflow as tf
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
k = knn_gscv.best_params_['n_neighbors']

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
score_knn = accuracy_score(y_test, y_pred_knn)

print("logistic_regression:", score_lr)
print("KNN:", score_knn)

# My method of finding accuracy:
# succZ=[]
# def pred(z):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=z)
#     lgr=LogisticRegression(solver='liblinear',multi_class='auto')
#     lgr.fit(X_train, y_train)
#
#     y_pred=lgr.predict(X_test)
#     count=0
#     match=0;unmatch=0;
#     for i in range(len(y_pred)):
#         count=count+1
#         if(y_pred[i]==y_test[i]):
#             match=match+1
#         else:
#             unmatch=unmatch+1
#
#
#     print(" Total:",count,"\n","Matched:",match,"\n","Unmatched:",unmatch)
#
#     percentage=(match/count)*100;
#
#     print(" Success:",percentage,"%");
#
#     if percentage==100:
#         succZ.append(z)
#         print("--------100% found----------")
#
#
#
# for z in range(1, 110):
#     print(z+1,":", pred(z))
#
#
# print("100% success when 'random_state' is either of these values:",succZ,"But why?")
