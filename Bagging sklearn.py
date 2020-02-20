# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:20:25 2020

@author: Lucky
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.svm import SVC
svm = SVC()
fit_svm = svm.fit(x_train, y_train)
print(fit_svm)

from sklearn.ensemble import BaggingClassifier
bag = BaggingClassifier(base_estimator = svm, n_estimators = 5)
fit_bag = bag.fit(x_train, y_train)
print(fit_bag)

bag_predict = bag.predict(x_test)
print(accuracy_score(bag_predict, y_test))