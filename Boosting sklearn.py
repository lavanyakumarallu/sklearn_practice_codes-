# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:31:20 2020

@author: Lucky
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset = load_iris()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.svm import SVC
svm = SVC(probability = True)
fit_svm = svm.fit(x_train, y_train)
print(fit_svm)

from sklearn.ensemble import AdaBoostClassifier

boost = AdaBoostClassifier(base_estimator = svm, n_estimators = 5, algorithm = 'SAMME')
fit_boost = boost.fit(x_train, y_train)
boost_predict = boost.predict(x_test)
print(fit_boost)

print(accuracy_score(boost_predict, y_test))