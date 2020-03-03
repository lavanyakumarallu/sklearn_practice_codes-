from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import Perceptron

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
ppn = Perceptron()
ppn.fit(x_train, y_train)
pred = ppn.predict(x_test)
print(accuracy_score(pred, y_test))
print(confusion_matrix(pred, y_test))