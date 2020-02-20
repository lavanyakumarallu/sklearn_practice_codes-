# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:18:43 2020

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
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
print(rf.fit(x_train, y_train))
y_predict = rf.predict(x_test)
print(accuracy_score(y_predict, y_test))
print(confusion_matrix(y_predict, y_test))