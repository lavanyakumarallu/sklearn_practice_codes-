# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:10:34 2020

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

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.fit_transform(x_test)



ppn = perceptron.Perceptron(eta0 = 0.1, random_state = 0, penalty = 'l2')
print(ppn)

print(ppn.fit(x_train_std,y_train))

test_predict = ppn.predict(x_test_std
                           )
train_predict = ppn.predict(x_train_std)
print(test_predict)
print(train_predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,train_predict))
print(accuracy_score(y_test, test_predict))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, test_predict))

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
print(lr.fit(x_train_std, y_train))

train_predict1 = lr.predict(x_train_std)
test_predict1 = lr.predict(x_test_std)

print(accuracy_score(y_train,train_predict1))
print(accuracy_score(y_test, test_predict1))
print(confusion_matrix(y_test, test_predict1))






