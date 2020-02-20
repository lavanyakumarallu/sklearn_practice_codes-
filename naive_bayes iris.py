# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:14:12 2020

@author: Lucky
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
gnb = GaussianNB()
print(gnb)
gnb.fit(x_train, y_train)

test_predict = gnb.predict(x_test)

print(accuracy_score(test_predict, y_test))
print(confusion_matrix(test_predict, y_test))