
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


# In[ ]:

#Setting the parameters for the Linear Regression Model Model


# # Example 1

# In[ ]:

#Default Values
param_lr = en.parameter_set_linear_regression()
print(param_lr)


# # Example 2

# In[ ]:

#Setting fit_intercept, rest are default values
param_lr = en.parameter_set_linear_regression(fit_intercept = [False])
print(param_lr)


# # Example 3

# In[ ]:

#Setting fit_intercept, normalize rest are default values
#Hyper parameter optimisation - fit_intercept
#Hyper parameter optimisation - normalize
param_lr = en.parameter_set_linear_regression(fit_intercept = [True, False], normalize = [True, False])
print(param_lr)

