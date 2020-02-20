# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:47:37 2020

@author: Lucky
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

dataset = load_iris()
x = dataset.data 
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
'''
knn = KNeighborsClassifier(n_neighbors = 10)
print(knn)
print(knn.fit(x_train, y_train))

x_predict = knn.predict(x_train)
y_predict = knn.predict(x_test)

print(accuracy_score(y_predict, y_test))
'''
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    x_predict = knn.predict(x_train)
    y_predict = knn.predict(x_test)
    print(k,"Accuracy is : " ,accuracy_score(y_predict, y_test))
    