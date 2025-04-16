#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:28:23 2025

@author: aime


"""

import numpy as np
import matplotlib as plt
import panda as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# 1) Importing and Extracting Data
dataset = pd.read_csv('data.csv')

X = dataset.iloc[:, 1:-1].to_numpy() # Extracting the independent variables
y = dataset.iloc[:, -1].to_numpy() # Extracting the dependent variable`

# 2) Data Preprocessing
 = PolynomialFeatures(degree = 2, include_bias = False).fit_transform(X)
# Transforming the independent variables to polynomial features



