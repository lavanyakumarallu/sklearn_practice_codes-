# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 10:05:42 2020

@author: Lucky
"""

from sklearn.datasets import load_iris

dataset = load_iris()
x = dataset.data
y = dataset.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

from sklearn.linear_model import Perceptron, LogisticRegression

ppn = Perceptron()
fit1 = ppn.fit(x_train, y_train)
print(fit1)

lr = LogisticRegression()
fit2 = lr.fit(x_train, y_train)
print(fit2)

from sklearn.svm import SVC
svm = SVC()
fit3 = svm.fit(x_train, y_train)
print(fit3)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
fit4 = dt.fit(x_train, y_train)
print(fit4)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
fit5 = knn.fit(x_train, y_train)
print(fit5)

from sklearn.ensemble import VotingClassifier
vote = VotingClassifier(estimators = [('ppn',ppn),('lr',lr),('svm',svm),
                                      ('dt',dt),('knn',knn)], voting = 'hard')
print(vote.fit(x_train, y_train))



            
y_ppn = ppn.predict(x_test)
y_lr = lr.predict(x_test)
y_svm = svm.predict(x_test)
y_dt = dt.predict(x_test)
y_knn = knn.predict(x_test)
y_vote = vote.predict(x_test)
from sklearn.metrics import accuracy_score
print('pn ' ,accuracy_score(y_ppn, y_test))
print('lr ', accuracy_score(y_lr, y_test))
print('svm ', accuracy_score(y_svm, y_test))
print('dt ', accuracy_score(y_dt, y_test))
print('knn ', accuracy_score(y_knn, y_test))
print(accuracy_score(y_vote, y_test))
