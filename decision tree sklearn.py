# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:19:57 2020

@author: Lucky
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

dataset = load_iris()
x = dataset.data
y = dataset.target

sc = StandardScaler()
x = sc.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "gini")
print(dt.fit(x_train, y_train))
y_predict = dt.predict(x_test)
print(accuracy_score(y_predict, y_test))

dt = DecisionTreeClassifier(criterion = "entropy")
print(dt.fit(x_train, y_train))
y_predict = dt.predict(x_test)
print(accuracy_score(y_predict, y_test))