# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:16:22 2020

@author: Lucky
"""

from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

dataset = load_iris()
x = dataset.data
y = dataset.target
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

print('linear')
svm  = SVC(kernel = 'linear')
print(svm)

print(svm.fit(x_train, y_train))

x_predict = svm.predict(x_train)
print(x_predict)
y_predict = svm.predict(x_test)
print(y_predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict, y_test))

print('rbf')
svm  = SVC(kernel = 'rbf')
print(svm)

print(svm.fit(x_train, y_train))

x_predict = svm.predict(x_train)
print(x_predict)
y_predict = svm.predict(x_test)
print(y_predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict, y_test))

print('poly')
svm  = SVC(kernel = 'poly')
print(svm)

print(svm.fit(x_train, y_train))

x_predict = svm.predict(x_train)
print(x_predict)
y_predict = svm.predict(x_test)
print(y_predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_predict, y_test))