
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

#Setting the parameters for the Multi Layer Perceptron Model (Number Of Layers = 3)
en.set_no_of_layers(3)


# # Example 1

# In[3]:

#Default Values
param_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = False)
print(param_mlp)


# # Example 2

# In[4]:

#Setting dim_layer, activation, rest are deafault values
param_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = False,                                                     dim_layer = [[32], [64], [1]],                                                    activation = [['sigmoid'], ['sigmoid'], ['relu']])
print(param_mlp)


# # Example 3

# In[5]:

#Setting dim_layer, activation, rest are default values
#Hyper parameter optimisation : dim_layer - Layer1 and Layer 2
#Hyper parameter optimisation : activation - Layer1 
param_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = True,                                                     dim_layer = [[32,64,128], [32,64], [1]],                                                    activation = [['sigmoid','relu'], ['sigmoid'], ['relu']],                                                    optimizer = 'sgd')
print(param_mlp)


# # Example 4

# In[6]:

#Setting dim_layer, activation, init_layer, rest are default values
#Hyper parameter optimisation : dim_layer - Layer1 
#Hyper parameter optimisation : activation - Layer1 
param_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = True,                                                     dim_layer = [[32,64,128], [64], [1]],                                                    activation = [['sigmoid','relu'], ['sigmoid'], ['relu']],                                                   init_layer = [['glorot_uniform'],['normal'],['glorot_uniform']])
print(param_mlp)


# # Example 5

# In[7]:

#Setting dim_layer, activation, init_layer, rest are default values
#Hyper parameter optimisation : dim_layer - Layer1 
#Hyper parameter optimisation : activation - Layer1 
#Hyper parameter optimisation : init_layer - Layer1 and Layer3
param_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = True,                                                     dim_layer = [[32,64,128], [64], [1]],                                                    activation = [['sigmoid','relu'], ['sigmoid'], ['relu']],                                                   init_layer = [['glorot_uniform','uniform'],                                                                 ['normal'], ['glorot_uniform','glorot_uniform']])
print(param_mlp)

