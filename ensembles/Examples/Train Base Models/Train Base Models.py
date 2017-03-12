
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


# In[2]:

#Training the base models


# # Example 1

# In[3]:

get_ipython().run_cell_magic('time', '', "Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)\ndata_test = en.data_import(Data, label_output='y')\nprint('Training Data',Data.shape)\nprint('Test Data',data_test.shape)\n\nen.metric_set('roc_auc_score')\n\n#Hyper Parameter Optimisation (max_depth and eta)\nparam_gb = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True, \\\n                                                eval_metric = ['auc'], objective = ['binary:logistic'], \\\n                                              max_depth = [5, 10, 15], eta = [0.1, 0.3, 0.5])\n\n#Setting max_depth, rest are default values\nparam_dt = en.parameter_set_decision_tree(max_depth = [6])\n\nen.train_base_models(['gradient_boosting','decision_tree'],[param_gb, param_dt], save_models = True)\n\n#Models saved as .pkl files\n[gb, dt] = en.get_base_models()\n\nprint('Gradient Boosting Model\\n', gb)\nprint('\\nDecision Tree Model\\n', dt)")


# # Example 2

# In[4]:

get_ipython().run_cell_magic('time', '', "Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)\ndata_test = en.data_import(Data, label_output='y',encode = 'binary')\nprint('Training Data',Data.shape)\nprint('Test Data',data_test.shape)\n\nen.metric_set('roc_auc_score')\n\nen.set_no_of_layers(4)\n\n#Setting penalty, rest are default values\nparam_lor = en.parameter_set_logistic_regression(penalty = ['l1'])\n\n#Setting fit_intercept, rest are default values\nparam_lr = en.parameter_set_linear_regression(fit_intercept = [False])\n\n#Setting dim_layer, activation, rest are deafault values\nparam_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = False, \\\n                                                    dim_layer = [[32], [64], [32], [1]], \\\n                                                   activation = [['sigmoid'], ['relu'], ['sigmoid'], ['relu']])\n\n#MLP does not work well with binary encode (changes to be made)\nen.train_base_models(['linear_regression','logistic_regression', 'multi_layer_perceptron'], \\\n                     [param_lr, param_lor, param_mlp])")


# # Example 3

# In[5]:

get_ipython().run_cell_magic('time', '', "Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)\ndata_test = en.data_import(Data, label_output='y', encode ='binary', split = True, stratify = False, split_size = 0.1)\nprint('Training Data',Data.shape)\nprint('Test Data',data_test.shape)\n\nen.metric_set('roc_auc_score')\n\n#Hyper Parameter Optimisation (gamma and eta)\nparam_gb = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True, \\\n                                                eval_metric = ['auc'], objective = ['binary:logistic'], \\\n                                                gamma = [0, 1, 3, 5, 7], eta = [0.1, 0.3], \\\n                                                max_depth = [5, 10, 15], colsample_bylevel = [0.1])\n\n#Setting max_depth, splitter, presort rest are default values\n#Hyper parameter optimisation - max_depth\n#Hyper parameter optimisation - splitter\nparam_dt_1 = en.parameter_set_decision_tree(max_depth = [6, 10, 12, 15], splitter = ['best', 'random'], \\\n                                          presort = [True])\n#Default Values\nparam_dt_2 = en.parameter_set_decision_tree()\n\nen.train_base_models(['decision_tree','decision_tree', 'gradient_boosting'], \\\n                     [param_dt_1, param_dt_2, param_gb])")


# # Example 4

# In[6]:

get_ipython().run_cell_magic('time', '', "Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)\ndata_test = en.data_import(Data, label_output='y')\nprint('Training Data',Data.shape)\nprint('Test Data',data_test.shape)\n\nen.metric_set('roc_auc_score')\n\nen.set_no_of_layers(3)\n\n#Hyper Parameter Optimisation (max_depth and eta)\nparam_gb = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True, \\\n                                                eval_metric = ['auc'], objective = ['binary:logistic'], \\\n                                              max_depth = [5, 10, 15], eta = [0.1, 0.3, 0.5])\n\n#Setting n_estimators, criterion, rest are default values\n#Hyper parameter optimisation - n_estimators\nparam_rf = en.parameter_set_random_forest(n_estimators = [6, 10, 12], criterion = ['entropy'])\n\n#Setting dim_layer, activation, rest are default values\n#Hyper parameter optimisation : dim_layer - Layer1 and Layer 2\n#Hyper parameter optimisation : activation - Layer1 \nparam_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = True, \\\n                                                    dim_layer = [[32,64,128], [32,64], [1]], \\\n                                                   activation = [['sigmoid','relu'], \\\n                                                                 ['sigmoid'], ['sigmoid','relu']], \\\n                                                   optimizer = 'sgd')\n\nen.train_base_models(['random_forest','multi_layer_perceptron', 'gradient_boosting'], \\\n                     [param_rf, param_mlp, param_gb])")


# # Example 5

# In[2]:

get_ipython().run_cell_magic('time', '', "Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)\ndata_test = en.data_import(Data, label_output='y')\nprint('Training Data',Data.shape)\nprint('Test Data',data_test.shape)\n\nen.metric_set('roc_auc_score')\n\nen.set_no_of_layers(3)\n\n#Hyper Parameter Optimisation (max_depth and eta)\nparam_gb_1 = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True, \\\n                                                eval_metric = ['auc'], objective = ['binary:logistic'], \\\n                                              max_depth = [5, 10, 15], eta = [0.1, 0.3, 0.5])\n\n#Hyper Parameter Optimisation (gamma and eta)\nparam_gb_2 = en.parameter_set_gradient_boosting(hyper_parameter_optimisation = True, \\\n                                                eval_metric = ['auc'], objective = ['binary:logistic'], \\\n                                                gamma = [0, 1, 3, 5, 7], eta = [0.1, 0.3], \\\n                                                max_depth = [5, 10, 15], colsample_bylevel = [0.1])\n\n\n#Setting max_depth, rest are default values\nparam_dt = en.parameter_set_decision_tree(max_depth = [6])\n\n#Setting max_depth, n_estimators, max_features, rest are default values\n#Hyper parameter optimisation - max_depth\n#Hyper parameter optimisation - n_estimators\nparam_rf = en.parameter_set_random_forest(max_depth = [6, 10, 12, 15], n_estimators = [10, 20, 30], \\\n                                          max_features = ['log2'])\n\n#Setting penalty, C, rest are default values\n#Hyper parameter optimisation - penalty\n#Hyper parameter optimisation - C\nparam_lor = en.parameter_set_logistic_regression(penalty = ['l1','l2'], C = [1.0, 2.0, 3.0, 5.0, 10.0])\n\n#Setting fit_intercept, rest are default values\nparam_lr = en.parameter_set_linear_regression(fit_intercept = [False])\n\n#Setting dim_layer, activation, rest are default values\n#Hyper parameter optimisation : dim_layer - Layer1 and Layer 2\n#Hyper parameter optimisation : activation - Layer1 and Layer 2\nparam_mlp = en.parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = True, \\\n                                                    dim_layer = [[32,64,128], [32,64], [1]], \\\n                                                   activation = [['sigmoid','relu'], \\\n                                                                 ['sigmoid'], ['sigmoid','relu']], \\\n                                                   optimizer = 'rmsprop')\n\n\n\nen.train_base_models(['random_forest','multi_layer_perceptron', 'gradient_boosting', \\\n                      'logistic_regression','linear_regression', 'decision_tree'], \\\n                     [param_rf, param_mlp, param_gb_1, param_lor, param_lr, param_dt])")


# In[ ]:



