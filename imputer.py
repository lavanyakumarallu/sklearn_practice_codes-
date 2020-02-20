# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 09:32:09 2020

@author: Lucky
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

data = pd.read_csv('F:\programs\python\int247\imputer.csv')
df = pd.DataFrame(data)
print(df)
# 'mean' , 'median', 'most_frequent'
si = SimpleImputer(missing_values=np.nan, strategy = 'mean')
fit = si.fit_transform(data)
print(fit)

si1 = SimpleImputer(missing_values=np.nan, strategy = 'median')
fit1 = si1.fit_transform(data)
print(fit1)

si2 = SimpleImputer(missing_values=np.nan, strategy = 'most_frequent')
fit2 = si2.fit_transform(data)
print(fit2)