# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 09:40:43 2020

@author: Lucky
"""

import numpy as np

n_classifier = 11
base_error = 0.25
k = n_classifier//2 + 1
print(k)

def fact(n):
    s = 1
    for i in range(1, n+1):
        s *= i
    return s

def combination(n, k):
    return fact(n)/(fact(k) * fact(n-k))

def ensemble_error(n, k, base_error):
    error = 0
    for i in range(k, n+1):
        error += combination(n, i) * (base_error)**i * (1 - base_error)**(n - i)
    return error

ens_error = ensemble_error(n_classifier, k, base_error)
print(ens_error)

base_error = np.arange(0, 1.01, 0.01)
print(base_error)
ens_error = ensemble_error(n_classifier, k, base_error)
print(ens_error)

import matplotlib.pyplot as plt
plt.plot(base_error, base_error, c = 'r')
plt.plot(base_error, ens_error)
plt.show()

