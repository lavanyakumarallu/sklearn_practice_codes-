# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:29:31 2020

@author: Lucky
"""

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,1,0.01)

def entrophy(x):
    return (-x*np.log2(x)-(1-x)*np.log2(1-x))

ent = []
for i in x:
    ent.append(entrophy(i))
def gini(p):
    return (1-(x**2+(1-x)**2))

giniindex = []
for i in x:
    giniindex.append(gini(i))

    
plt.plot(x,ent, c='yellow')
plt.plot(x,giniindex)
plt.show()