# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 09:59:38 2020

@author: Lucky
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
x = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
y = np.array([0,0,0,0,1,1,1,1])

gnd = GaussianNB()
GaussianNB(priors = None)
gnd.fit(x,y)
print(gnd.predict([[1,1,0]]))
print(gnd.predict([[2,0,0]]))
z = gnd.predict(x)
print('misclassified samples:%d'%(z!=y).sum())
from sklearn.metrics import accuracy_score
print(accuracy_score(z,y))
