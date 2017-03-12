
# coding: utf-8

# In[ ]:

import ensembles as en
import pandas as pd
import numpy as np
import xgboost as xgb
import category_encoders as ce
from sklearn import datasets, linear_model, preprocessing, grid_search
from sklearn.preprocessing import Imputer, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.externals import joblib
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, log_loss, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.cross_validation import train_test_split
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from functools import partial
np.random.seed(1338)


# In[2]:

#Setting the parameters for the Decision Tree Model (Number Of Layers = 3)


# # Example 1

# In[3]:

#Default Values
param_dt = en.parameter_set_decision_tree()
print(param_dt)


# # Example 2

# In[4]:

#Setting max_depth, rest are default values
param_dt = en.parameter_set_decision_tree(max_depth = [6])
print(param_dt)


# # Example 3

# In[5]:

#Setting max_depth, criterion, rest are default values
#Hyper parameter optimisation - max_depth
param_dt = en.parameter_set_decision_tree(max_depth = [6, 10, 12], criterion = ['entropy'])
print(param_dt)


# # Example 4

# In[6]:

#Setting max_depth, splitter, rest are default values
#Hyper parameter optimisation - max_depth
#Hyper parameter optimisation - splitter
param_dt = en.parameter_set_decision_tree(max_depth = [6, 10, 12], splitter = ['best', 'random'])
print(param_dt)


# # Example 5

# In[7]:

#Setting max_depth, splitter, presort rest are default values
#Hyper parameter optimisation - max_depth
#Hyper parameter optimisation - splitter
param_dt = en.parameter_set_decision_tree(max_depth = [6, 10, 12, 15], splitter = ['best', 'random'],                                           presort = True)
print(param_dt)

