
# coding: utf-8

# In[1]:

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


# # Example 1

# In[2]:

Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)
data_test = en.data_import(Data, label_output='y')
print('Training Data',Data.shape)
print('Test Data',data_test.shape)


# # Example 2

# In[3]:

Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)
data_test = en.data_import(Data, label_output='y',encode = 'binary')
print('Training Data',Data.shape)
print('Test Data',data_test.shape)


# # Example 3

# In[4]:

Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)
en.data_import(Data, label_output='y', encode = None, split = False)
print('Training Data',Data.shape)


# # Example 4

# In[5]:

Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)
data_test = en.data_import(Data, label_output='y', encode ='sum', split = True, stratify = False, split_size = 0.1)
print('Training Data',Data.shape)
print('Test Data',data_test.shape)


# # Example 5

# In[6]:

Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)
en.data_import(Data, label_output='y', encode = 'binary', split = False)
print('Training Data',Data.shape)

