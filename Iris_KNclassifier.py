# import dependencies
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# load datast and get features and targets
iris_flower = load_iris()
X = iris_flower.data
y = iris_flower.target

# find best value for n_neighbors by cross_checking
knn2 = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 50)}
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
knn_gscv.fit(X, y)
k = knn_gscv.best_params_['n_neighbors']
print(k)

# predict
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)
result = knn.predict([[2, 4, 3, 1], [4, 6, 5, 3]])
print(result, iris_flower.target_names[result])
