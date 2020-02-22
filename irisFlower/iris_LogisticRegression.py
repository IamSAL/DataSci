# import dependencies
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# load datast and get features and targets
iris_flower = load_iris()
X = iris_flower.data
y = iris_flower.target

# predict
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr.fit(X, y)
result = lr.predict([[2, 4, 3, 1], [4, 6, 5, 3]])
print(result, iris_flower.target_names[result])
