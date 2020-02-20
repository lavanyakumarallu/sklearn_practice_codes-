# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:40:18 2020

@author: Lucky
"""

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

dataset = load_iris()
x = dataset.data
y = dataset.target
print(x)
print(y)
print(x.shape)
print(y.shape)


import pandas as pd
print(pd.DataFrame(y).nunique())

plt.scatter(x[:,0],x[:,1],c = y)
plt.title('feature [0,1]')
plt.show()
plt.scatter(x[:,1],x[:,2],c = y)
plt.title('feature [1,2]')
plt.show()
plt.scatter(x[:,2],x[:,3],c = y)
plt.title('feature [2,3]')
plt.show()
plt.scatter(x[:,3],x[:,1],c = y)
plt.title('feature [3,1]')
plt.show()


from sklearn.linear_model import perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

ppn = perceptron.Perceptron(eta0 = 0.1, random_state = 0)
print(ppn)

print(ppn.fit(x_train,y_train))

test_predict = ppn.predict(x_test)
train_predict = ppn.predict(x_train)
print(test_predict)
print(train_predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,train_predict))
print(accuracy_score(y_test, test_predict))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, test_predict))

x1 = x[:,2:]
print(x1.shape)

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size = 0.25, random_state = 0)
print(x1_train.shape)
print(x1_test.shape)

print(ppn.fit(x1_train, y1_train))

test1_predict = ppn.predict(x1_test)
train1_predict = ppn.predict(x1_train)

print(accuracy_score(y_train,train1_predict))
print(accuracy_score(y_test, test1_predict))

print(accuracy_score(y_train, train1_predict))
print(accuracy_score(y_test, test1_predict))

print(confusion_matrix(y_test, test1_predict))




