
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

#Setting the parameters for the Gradient Boosting Model


# # Example 1

# In[3]:

#Default Values
param_gb = en.parameter_set_gradient_boosting(eval_metric = ['auc'], objective = ['binary:logistic'])
print(param_gb)


# # Example 2

# In[4]:

#Changing max_depth and eta
param_gb = en.parameter_set_gradient_boosting(eval_metric = ['auc'], objective = ['binary:logistic'],                                                 max_depth = [10], eta = [0.5])
print(param_gb)


# # Example 3

# In[5]:

#Hyper Parameter Optimisation (max_depth and eta)
param_gb = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True,                                                 eval_metric = ['auc'], objective = ['binary:logistic'],                                               max_depth = [5, 10, 15], eta = [0.1, 0.3, 0.5])
print(param_gb)


# # Example 4

# In[6]:

#Hyper Parameter Optimisation (gamma and eta)
param_gb = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True,                                                 eval_metric = ['auc'], objective = ['binary:logistic'],                                                 gamma = [0, 1, 3, 5, 7], eta = [0.1, 0.3])
print(param_gb)


# # Example 5

# In[7]:

#Hyper Parameter Optimisation (gamma and eta)
param_gb = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True,                                                 eval_metric = ['auc'], objective = ['binary:logistic'],                                                 gamma = [0, 1, 3, 5, 7], eta = [0.1, 0.3],                                                 max_depth = [5, 10, 15], colsample_bylevel = [0.1])
print(param_gb)

