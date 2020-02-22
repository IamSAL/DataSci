#import dependencies
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

#load datast and get features and targets
iris_flower=load_iris()
X = iris_flower.data
y = iris_flower.target

#for easy management of data
set = np.array([[3, 4, 3, 1], [4, 6, 5, 3]])
res=[]

#check 1<n_neighbors<=30
for n in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X, y)
    result = knn.predict(set)
    ress=(set[0], set[1], result, iris_flower.target_names[result])
    res.append(ress) #storing for better printing and later use


#print for the list
for i in range(len(res)):
    print('-----------', i, '----------')
    for j in range(len(res[i])):#reached list,now print each value by looping as it contains arrays
        z=res[i]
        print(z[j],'\b')
    print('\n')

