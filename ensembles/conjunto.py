
# coding: utf-8

# In[1]:

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


# # Number Of Layers (MLP)

# In[2]:

def set_no_of_layers(number):
    
    global no_of_layers
    
    no_of_layers = number


# # Evaluation Metrics

# In[3]:

def metric_set(metric):
    
    global metric_score
    global metric_grid_search
    metric_functions = {'roc_auc_score' : [roc_auc_score, 'roc_auc'], 'average_precision_score' : 
                        [average_precision_score, 'average_precision'], 'f1_score' : [f1_score, 'f1'],
                        'log_loss' : [log_loss, 'log_loss'], 'accuracy_score' : [accuracy_score, 'accuracy'],
                        'mean_absolute_error' : [mean_absolute_error,'mean_absolute_error'],
                        'mean_squared_error':[mean_squared_error, 'mean_squared_error'],
                        'r2_score' : [r2_score, 'r2']
                        }
    
    metric_score = metric_functions[metric][0]
    metric_grid_search = metric_functions[metric][1]


# # Getting the data

# In[4]:

def data_import(data, label_output, encode = None, split = True, stratify = True, split_size = 0.3):
    
    global Data
    Data = data
    
    #Data = Data.dropna()
    #Data = data.fillna(data.mean())
    #Data = data.interpolate()
    #Data = data.fillna(data.median())
    #(interpolate)methods = {‘linear’, ‘time’, ‘index’, ‘values’, ‘nearest’, ‘zero’,
    #‘slinear’, ‘quadratic’, ‘cubic’, ‘barycentric’, ‘krogh’, ‘polynomial’, ‘spline’, ‘piecewise_polynomial’,\
    #‘from_derivatives’, ‘pchip’, ‘akima’}
    
    #Reading the data, into a Data Frame.
    global target_label
    target_label = label_output

    #Selcting the columns of string data type
    names = data.select_dtypes(include = ['object'])
    
    #Converting string categorical variables to integer categorical variables.
    label_encode(names.columns.tolist())
    
    if(target_label in names):
        
        columns = names.drop([target_label],axis=1).columns.tolist()
        
    else:
        
        columns = names
        
    #Data will be encoded to the form that the user enters
    encoding = {'binary':binary_encode,'hashing':hashing_encode,'backward_difference'
               :backward_difference_encode,'helmert':helmert_encode,'polynomial':
               polynomial_encode,'sum':sum_encode,'label':label_encode}
    
    if(encode != None):
        
        #Once the above encoding techniques has been selected by the user, 
        #the appropriate encoding function is called
        encoding[encode](columns)

        
    #This function intializes the dataframes that will be used later in the program
    #data_initialize()
    
    #Splitting the data into to train and test sets, according to user preference
    if(split == True):
        
        test_data = data_split(stratify,split_size)
        return test_data


# # Data for ensembling (Training)

# In[5]:

#The dataframes will be used in the training phase of the ensemble models
def second_level_train_data(predict_list, cross_val_X, cross_val_Y):
    
    #Converting the list of predictions into a dataframe, which will be used to train the stacking model.
    global stack_X
    stack_X = pd.DataFrame()
    stack_X = stack_X.append(build_data_frame(predict_list))
    
    #Building a list that contains all the raw features, used as cross validation data for the base models.
    global raw_features_X
    raw_features_X = pd.DataFrame()
    raw_features_X = raw_features_X.append(cross_val_X,ignore_index=True)
    
    #The data frame will contain the predictions and raw features  of the base models, for training the blending
    #model
    global blend_X
    blend_X = pd.DataFrame()
    blend_X = pd.concat([raw_features_X, stack_X], axis = 1, ignore_index = True)
    
    #Storing the cross validation dataset labels in the variable stack_Y, 
    #which will be used later to train the stacking and blending models.
    global stack_Y
    stack_Y = cross_val_Y  


# # Data for ensembling (Testing)

# In[6]:

#The dataframes will be used in the testing phase of the ensemble models
def second_level_test_data(predict_list, test_X, test_Y):
    
    #Converting the list of predictions into a dataframe, which will be used to test the stacking model.
    global test_stack_X
    test_stack_X = pd.DataFrame()
    test_stack_X = test_stack_X.append(build_data_frame(predict_list))
    
    #Building a list that contains all the raw features, used as test data for the base models.
    global test_raw_features_X
    test_raw_features_X = pd.DataFrame()
    test_raw_features_X = test_raw_features_X.append(test_X,ignore_index=True)
    
    #The data frame will contain the predictions and raw features of the base models, for testing the blending
    #model
    global test_blend_X
    test_blend_X = pd.DataFrame()
    test_blend_X = pd.concat([test_raw_features_X, test_stack_X], axis = 1, ignore_index = True)
    
    #Storing the cross validation dataset labels in the variable stack_Y, 
    #which will be used later to test the stacking and blending models.
    global test_stack_Y
    test_stack_Y = test_Y  


# # Label Encoding

# In[7]:

#Function that encodes the string values to numerical values.
def label_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values.
    encoder = ce.OrdinalEncoder(cols = column_names, verbose = 1)
    Data = encoder.fit_transform(Data)


# # Binary Encoding

# In[8]:

def binary_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values, using binary method.
    encoder = ce.BinaryEncoder(cols = column_names, verbose = 1)
    Data = encoder.fit_transform(Data)


# # Hashing Encoding

# In[9]:

def hashing_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values, using hashing method.
    encoder = ce.HashingEncoder(cols = column_names, verbose = 1, n_components = 128)
    Data = encoder.fit_transform(Data)


# # Backward Difference Encoding

# In[10]:

def backward_difference_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values, using backward difference method.
    encoder = ce.BackwardDifferenceEncoder(cols = column_names, verbose = 1)
    Data = encoder.fit_transform(Data)


# # Helmert Encoding

# In[11]:

def helmert_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values, using helmert method.
    encoder = ce.HelmertEncoder(cols = column_names, verbose = 1)
    Data = encoder.fit_transform(Data)


# # Sum Encoding

# In[12]:

def sum_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values, using sum method.
    encoder = ce.SumEncoder(cols = column_names, verbose = 1)
    Data = encoder.fit_transform(Data)  


# # Polynomial Encoding

# In[13]:

def polynomial_encode(column_names):
    
    global Data
    #Encoding the data, encoding the string values into numerical values, using polynomial method.
    encoder = ce.PolynomialEncoder(cols = column_names, verbose = 1)
    Data = encoder.fit_transform(Data)


# In[14]:

#Splitting the data into training and testing datasets
def data_split(stratify, split_size):
    
    global Data
    
    #Stratified Split
    if(stratify == True):
        Data, test = train_test_split(Data, test_size = split_size, stratify = Data[target_label],random_state = 0)
        
    #Random Split
    else:
        Data, test = train_test_split(Data, test_size = split_size,random_state = 0) 
        
    return test


# In[15]:

#This function is used to convert the predictions of the base models (numpy array) into a DataFrame.
def build_data_frame(data):
    
    data_frame = pd.DataFrame(data).T
    return data_frame


# # Gradient Boosting (XGBoost)

# In[16]:

#Trains the Gradient Boosting model.
def train_gradient_boosting(train_X, train_Y, parameter_gradient_boosting):

    #Hyperopt procedure, train the model with optimal paramter values
    if(parameter_gradient_boosting['hyper_parameter_optimisation'] == True):
        
        model = gradient_boosting_parameter_optimisation(train_X, train_Y, parameter_gradient_boosting,                                              objective_gradient_boosting)
        return model
    
    #Train the model with the parameter values entered by the user, no need to find otimal values
    else:
        
        dtrain = xgb.DMatrix(train_X, label = train_Y)
        del parameter_gradient_boosting['hyper_parameter_optimisation']
        model = xgb.train(parameter_gradient_boosting, dtrain)
        return model  


# In[17]:

#Defining the parameters for the XGBoost (Gradient Boosting) Algorithm.
def parameter_set_gradient_boosting(hyper_parameter_optimisation = False, eval_metric = None, booster = ['gbtree'],                                    silent = [0], eta = [0.3], gamma = [0], max_depth = [6],                                    min_child_weight = [1], max_delta_step = [0], subsample = [1],                                    colsample_bytree = [1], colsample_bylevel = [1], lambda_xgb = [1], alpha = [0],                                    tree_method = ['auto'], sketch_eps = [0.03], scale_pos_weight = [0],                                    lambda_bias = [0], objective = ['reg:linear'], base_score = [0.5],                                    num_class = None):

    parameter_gradient_boosting = {}
    #This variable will be used to check if the user wants to perform hyper parameter optimisation.
    parameter_gradient_boosting['hyper_parameter_optimisation'] = hyper_parameter_optimisation
    
    #Setting objective and seed
    parameter_gradient_boosting['objective'] = objective[0]
    parameter_gradient_boosting['seed'] = 0
    
    if(num_class != None):
        
        parameter_gradient_boosting['num_class'] = num_class
    
    #If hyper parameter optimisation is false, we unlist the default values and/or the values that the user enters 
    #in the form of a list. Values have to be entered by the user in the form of a list, for hyper parameter 
    #optimisation = False, these values will be unlisted below
    #Ex : booster = ['gbtree'](default value) becomes booster = 'gbtree'
    #This is done beacuse for training the model, the model does not accept list type values
    if(parameter_gradient_boosting['hyper_parameter_optimisation'] == False):
        
        #Setting the parameters for the Booster, list values are unlisted (E.x - booster[0])
        parameter_gradient_boosting['booster'] = booster[0]
        parameter_gradient_boosting['eval_metric'] = eval_metric[0]
        parameter_gradient_boosting['eta'] = eta[0]
        parameter_gradient_boosting['gamma'] = gamma[0]
        parameter_gradient_boosting['max_depth'] = max_depth[0]
        parameter_gradient_boosting['min_child_weight'] = min_child_weight[0]
        parameter_gradient_boosting['max_delta_step'] = max_delta_step[0]
        parameter_gradient_boosting['subsample'] = subsample[0]
        parameter_gradient_boosting['colsample_bytree'] = colsample_bytree[0]
        parameter_gradient_boosting['colsample_bylevel'] = colsample_bylevel[0]
        parameter_gradient_boosting['base_score'] = base_score[0]
        parameter_gradient_boosting['lambda_bias'] = lambda_bias[0]
        parameter_gradient_boosting['alpha'] = alpha[0]
        parameter_gradient_boosting['tree_method'] = tree_method[0]
        parameter_gradient_boosting['sketch_eps'] = sketch_eps[0]
        parameter_gradient_boosting['scale_pos_weigth'] = scale_pos_weight[0]
        parameter_gradient_boosting['lambda'] = lambda_xgb[0]
        
    else:
        
        #Setting parameters for the Booster which will be optimized later using hyperopt.
        #The user can enter a list of values that he wants to optimize
        parameter_gradient_boosting['booster'] = booster
        parameter_gradient_boosting['eval_metric'] = eval_metric
        parameter_gradient_boosting['eta'] = eta
        parameter_gradient_boosting['gamma'] = gamma
        parameter_gradient_boosting['max_depth'] = max_depth
        parameter_gradient_boosting['min_child_weight'] = min_child_weight
        parameter_gradient_boosting['max_delta_step'] = max_delta_step
        parameter_gradient_boosting['subsample'] = subsample
        parameter_gradient_boosting['colsample_bytree'] = colsample_bytree
        parameter_gradient_boosting['colsample_bylevel'] = colsample_bylevel
        parameter_gradient_boosting['base_score'] = base_score
        parameter_gradient_boosting['lambda_bias'] = lambda_bias
        parameter_gradient_boosting['alpha'] = alpha
        parameter_gradient_boosting['tree_method'] = tree_method
        parameter_gradient_boosting['sketch_eps'] = sketch_eps
        parameter_gradient_boosting['scale_pos_weigth'] = scale_pos_weight
        parameter_gradient_boosting['lambda'] = lambda_xgb
        
    return parameter_gradient_boosting


# In[18]:

#Using the loss values, this function picks the optimum parameter values. These values will be used 
#for training the model
def gradient_boosting_parameter_optimisation(train_X, train_Y, parameter_gradient_boosting,obj):
    
    space_gradient_boosting = assign_space_gradient_boosting(parameter_gradient_boosting)
    trials = Trials()
    
    #Best is used to otmize the objective function
    best = fmin(fn = partial(obj, data_X = train_X, data_Y = train_Y                             , parameter_gradient_boosting = parameter_gradient_boosting),
    space = space_gradient_boosting,
    algo = tpe.suggest,
    max_evals = 100,
    trials = trials)
    
    optimal_param={}
    #Best is a dictionary that contains the indices of the optimal parameter values.
    #The following for loop uses these indices to obtain the parameter values, these values are stored in a
    #dictionary - optimal_param
    for key in best:
        optimal_param[key] = parameter_gradient_boosting[key][best[key]]
        
    optimal_param['objective'] = parameter_gradient_boosting['objective']
    optimal_param['eval_metric'] = parameter_gradient_boosting['eval_metric']
    optimal_param['seed'] = parameter_gradient_boosting['seed']
    
    #Training the model with the optimal parameter values
    dtrain = xgb.DMatrix(train_X, label = train_Y)
    model = xgb.train(optimal_param, dtrain)
    return model


# In[19]:

#This function calculates the loss for different parameter values and is used to determine the most optimum 
#parameter values
def objective_gradient_boosting(space_gradient_boosting, data_X, data_Y, parameter_gradient_boosting):
    
    #Gradient Boosting (XGBoost)
    param = {}
    #Setting Parameters for the Booster
    param['booster'] = space_gradient_boosting['booster']
    param['objective'] = parameter_gradient_boosting['objective']
    param['eval_metric'] = parameter_gradient_boosting['eval_metric']
    param['eta'] = space_gradient_boosting['eta']
    param['gamma'] = space_gradient_boosting['gamma']
    param['max_depth'] = space_gradient_boosting['max_depth']
    param['min_child_weight'] = space_gradient_boosting['min_child_weight']
    param['max_delta_step'] = space_gradient_boosting['max_delta_step']
    param['subsample'] = space_gradient_boosting['subsample']
    param['colsample_bytree'] = space_gradient_boosting['colsample_bytree']
    param['colsample_bylevel'] = space_gradient_boosting['colsample_bylevel']
    param['alpha'] = space_gradient_boosting['alpha']
    param['scale_pos_weigth'] = space_gradient_boosting['scale_pos_weigth']
    param['base_score'] = space_gradient_boosting['base_score']
    param['lambda_bias'] = space_gradient_boosting['lambda_bias']
    param['lambda'] = space_gradient_boosting['lambda']
    param['tree_method'] = space_gradient_boosting['tree_method']
    
    
    metric_list = list()

    #Performing cross validation.
    skf=StratifiedKFold(data_Y, n_folds=3,random_state=0)
    for train_index, cross_val_index in skf:
        
        model = xgb.Booster()
        
        xgb_train_X, xgb_cross_val_X = data_X.iloc[train_index],data_X.iloc[cross_val_index]
        
        xgb_train_Y, xgb_cross_val_Y = data_Y.iloc[train_index],data_Y.iloc[cross_val_index]
        
        dtrain = xgb.DMatrix(xgb_train_X, label = xgb_train_Y)
        model = xgb.train(param, dtrain)
        
        predicted_values = model.predict(xgb.DMatrix(xgb_cross_val_X, label = xgb_cross_val_Y))
        
        if(metric_grid_search in ['f1', 'log_loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error', 'r2']):
            
            predictions = predicted_values >= 0.5
            predictions.astype(int)
            metric_list.append(metric_score(xgb_cross_val_Y,predictions))
            
        else :
            
            metric_list.append(metric_score(xgb_cross_val_Y,predicted_values))
            
        
    
    #Calculating the AUC and returning the loss, which will be minimised by selecting the optimum parameters.
    metric = np.mean(metric_list)
    return{'loss':1-metric, 'status': STATUS_OK }


# In[20]:

#Assigning the values of the XGBoost parameters that need to be checked, for minimizing the objective (loss).
#The values that give the most optimum results will be picked to train the model.
def assign_space_gradient_boosting(parameter_gradient_boosting):
    

    space_gradient_boosting ={
        
        'booster': hp.choice('booster', parameter_gradient_boosting['booster']),
        
        'eta': hp.choice('eta', parameter_gradient_boosting['eta']),
        
        'gamma': hp.choice('gamma', parameter_gradient_boosting['gamma']),
        
        'max_depth': hp.choice('max_depth', parameter_gradient_boosting['max_depth']),
        
        'min_child_weight': hp.choice('min_child_weight', parameter_gradient_boosting['min_child_weight']),
        
        'max_delta_step': hp.choice('max_delta_step', parameter_gradient_boosting['max_delta_step']),
        
        'subsample': hp.choice('subsample', parameter_gradient_boosting['subsample']),
        
        'colsample_bytree': hp.choice('colsample_bytree', parameter_gradient_boosting['colsample_bytree']),
        
        'colsample_bylevel': hp.choice('colsample_bylevel', parameter_gradient_boosting['colsample_bylevel']),
        
        'alpha': hp.choice('alpha', parameter_gradient_boosting['alpha']),
        
        'scale_pos_weigth': hp.choice('scale_pos_weigth', parameter_gradient_boosting['scale_pos_weigth']),
        
        'base_score': hp.choice('base_score', parameter_gradient_boosting['base_score']),
        
        'lambda_bias': hp.choice('lambda_bias', parameter_gradient_boosting['lambda_bias']),
        
        'lambda': hp.choice('lambda', parameter_gradient_boosting['lambda']),
        
        'tree_method': hp.choice('tree_method', parameter_gradient_boosting['tree_method'])
        
    }
    
    return space_gradient_boosting


# In[21]:

def predict_gradient_boosting(data_X, data_Y, gradient_boosting):
    
    predicted_values = gradient_boosting.predict(xgb.DMatrix(data_X, label = data_Y))
            
    metric = metric_score(data_Y,predicted_values)

    return [metric,predicted_values]


# # Multi Layer Perceptron

# In[22]:

#Trains the Gradient Boosting model.
def train_multi_layer_perceptron(train_X, train_Y, parameter_multi_layer_perceptron):
    
    global no_of_layers

    #Hyperopt procedure, train the model with optimal paramter values
    if(parameter_multi_layer_perceptron['hyper_parameter_optimisation'] == True):
        
        model = multi_layer_perceptron_parameter_optimisation(train_X, train_Y, parameter_multi_layer_perceptron,                                              objective_multi_layer_perceptron)
        return model
    
    #Train the model with the parameter values entered by the user, no need to find otimal values
    else:
        
        del parameter_multi_layer_perceptron['hyper_parameter_optimisation']
        train_X = StandardScaler().fit_transform(train_X)
        #train_X = train_X.as_matrix()
        #train_Y = train_Y.as_matrix()
        model = Sequential()
        
        for i in range(no_of_layers):
            
            if(i==0):
                
                model.add(Dense(output_dim = parameter_multi_layer_perceptron['dim_layer'+str(i)],                                 input_dim = train_X.shape[1],                                 init = parameter_multi_layer_perceptron['init_layer'+str(i)], 
                                activation = parameter_multi_layer_perceptron['activation_layer'+str(i)], \
                                weights = parameter_multi_layer_perceptron['weights'+str(i)], \
                                W_regularizer = parameter_multi_layer_perceptron['W_regularizer'+str(i)], \
                                b_regularizer = parameter_multi_layer_perceptron['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                parameter_multi_layer_perceptron['activity_regularizer'+str(i)], \
                                W_constraint = parameter_multi_layer_perceptron['W_constraint'+str(i)], \
                                b_constraint = parameter_multi_layer_perceptron['b_constraint'+str(i)], \
                                bias = parameter_multi_layer_perceptron['bias'+str(i)]))
                
                if(parameter_multi_layer_perceptron['dropout'+str(i)] != 0):
                    model.add(Dropout(parameter_multi_layer_perceptron['dropout'+str(i)]))
                    
            elif(i==no_of_layers-1):
                
                model.add(Dense(output_dim = 1,                                 input_dim = train_X.shape[1],                                 init = parameter_multi_layer_perceptron['init_layer'+str(i)], 
                                activation = parameter_multi_layer_perceptron['activation_layer'+str(i)], \
                                weights = parameter_multi_layer_perceptron['weights'+str(i)], \
                                W_regularizer = parameter_multi_layer_perceptron['W_regularizer'+str(i)], \
                                b_regularizer = parameter_multi_layer_perceptron['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                parameter_multi_layer_perceptron['activity_regularizer'+str(i)], \
                                W_constraint = parameter_multi_layer_perceptron['W_constraint'+str(i)], \
                                b_constraint = parameter_multi_layer_perceptron['b_constraint'+str(i)], \
                                bias = parameter_multi_layer_perceptron['bias'+str(i)]))
                
            else:
                
                model.add(Dense(output_dim = parameter_multi_layer_perceptron['dim_layer'+str(i)],                                 input_dim = train_X.shape[1],                                 init = parameter_multi_layer_perceptron['init_layer'+str(i)], 
                                activation = parameter_multi_layer_perceptron['activation_layer'+str(i)], \
                                weights = parameter_multi_layer_perceptron['weights'+str(i)], \
                                W_regularizer = parameter_multi_layer_perceptron['W_regularizer'+str(i)], \
                                b_regularizer = parameter_multi_layer_perceptron['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                parameter_multi_layer_perceptron['activity_regularizer'+str(i)], \
                                W_constraint = parameter_multi_layer_perceptron['W_constraint'+str(i)], \
                                b_constraint = parameter_multi_layer_perceptron['b_constraint'+str(i)], \
                                bias = parameter_multi_layer_perceptron['bias'+str(i)]))
                
                if(parameter_multi_layer_perceptron['dropout'+str(i)] != 0):
                    model.add(Dropout(parameter_multi_layer_perceptron['dropout'+str(i)]))
                    
        model.compile(optimizer = parameter_multi_layer_perceptron['optimizer'],                       loss = 'binary_crossentropy',metrics = ['accuracy'])
        model.fit(train_X, train_Y, nb_epoch = 10, batch_size = 128, verbose=1)
    

        return model 


# In[23]:

def parameter_set_multi_layer_perceptron(hyper_parameter_optimisation = False, optimizer = ['rmsprop'],                                          init_layer = [], dim_layer = [],                                          activation = [], dropout = [], weights = [],                                          W_regularizer = [], b_regularizer = [],                                          activity_regularizer = [], W_constraint = [],                                          b_constraint = [], bias = []):
    
    global no_of_layers
    
    parameter_multi_layer_perceptron = {}
    #This variable will be used to check if the user wants to perform hyper parameter optimisation.
    parameter_multi_layer_perceptron['hyper_parameter_optimisation'] = hyper_parameter_optimisation
    
    if (parameter_multi_layer_perceptron['hyper_parameter_optimisation'] == 'Default'):
        
        x = 1
    
    else:
        
        x = no_of_layers
        
        
    if (init_layer == []):
        
        init_layer = [['glorot_uniform']] * x
        
    if (dim_layer == []):
        
        dim_layer = [[32]] * x
            
    if (activation == []):
        
        activation = [['sigmoid']] * x
        
    if (dropout == []):
        
        dropout = [[0]] * x
        
    if (weights == []):
        
        weights = [[None]] * x
        
    if (W_regularizer == []):
        
        W_regularizer = [[None]] * x
        
    if (b_regularizer == []):
        
        b_regularizer = [[None]] * x
        
    if (activity_regularizer == []):
        
        activity_regularizer = [[None]] * x
        
    if (W_constraint == []):
        
        W_constraint = [[None]] * x
        
    if (b_constraint == []):
        
        b_constraint = [[None]] * x
        
    if (bias == []):
        
        bias= [[True]] * x
    
    
    
    for i in range(no_of_layers):
        
        if(parameter_multi_layer_perceptron['hyper_parameter_optimisation'] == False):
            
            parameter_multi_layer_perceptron['dim_layer'+str(i)] = dim_layer[i][0]
            parameter_multi_layer_perceptron['activation_layer'+str(i)] = activation[i][0]
            parameter_multi_layer_perceptron['init_layer'+str(i)] = init_layer[i][0]
            parameter_multi_layer_perceptron['dropout'+str(i)] = dropout[i][0]
            parameter_multi_layer_perceptron['weights'+str(i)] = weights[i][0]
            parameter_multi_layer_perceptron['W_regularizer'+str(i)] = W_regularizer[i][0]
            parameter_multi_layer_perceptron['b_regularizer'+str(i)] = b_regularizer[i][0]
            parameter_multi_layer_perceptron['activity_regularizer'+str(i)] = activity_regularizer[i][0]
            parameter_multi_layer_perceptron['W_constraint'+str(i)] = W_constraint[i][0]
            parameter_multi_layer_perceptron['b_constraint'+str(i)] = b_constraint[i][0]
            parameter_multi_layer_perceptron['bias'+str(i)] = bias[i][0]
            parameter_multi_layer_perceptron['optimizer'] = optimizer[0]
            parameter_multi_layer_perceptron['dim_layer'+str(no_of_layers-1)] = 1
            parameter_multi_layer_perceptron['dropout'+str(no_of_layers-1)] = 0
               
        else:
            
            parameter_multi_layer_perceptron['dim_layer'+str(i)] = dim_layer[i]
            parameter_multi_layer_perceptron['activation_layer'+str(i)] = activation[i]
            parameter_multi_layer_perceptron['init_layer'+str(i)] = init_layer[i]
            parameter_multi_layer_perceptron['dropout'+str(i)] = dropout[i]
            parameter_multi_layer_perceptron['weights'+str(i)] = weights[i]
            parameter_multi_layer_perceptron['W_regularizer'+str(i)] = W_regularizer[i]
            parameter_multi_layer_perceptron['b_regularizer'+str(i)] = b_regularizer[i]
            parameter_multi_layer_perceptron['activity_regularizer'+str(i)] = activity_regularizer[i]
            parameter_multi_layer_perceptron['W_constraint'+str(i)] = W_constraint[i]
            parameter_multi_layer_perceptron['b_constraint'+str(i)] = b_constraint[i]
            parameter_multi_layer_perceptron['bias'+str(i)] = bias[i]
            parameter_multi_layer_perceptron['optimizer'] = optimizer
            parameter_multi_layer_perceptron['dim_layer'+str(no_of_layers-1)] = [1]
            parameter_multi_layer_perceptron['dropout'+str(no_of_layers-1)] = [0]
              
    return parameter_multi_layer_perceptron


# In[24]:

#Using the loss values, this function picks the optimum parameter values. These values will be used 
#for training the model
def multi_layer_perceptron_parameter_optimisation(train_X, train_Y, parameter_multi_layer_perceptron,obj):
    
    global no_of_layers
    
    space_multi_layer_perceptron = assign_space_multi_layer_perceptron(parameter_multi_layer_perceptron)
    trials = Trials()
    
    #Best is used to otmize the objective function
    best = fmin(fn = partial(obj, data_X = train_X, data_Y = train_Y                             , parameter_multi_layer_perceptron = parameter_multi_layer_perceptron),
    space = space_multi_layer_perceptron,
    algo = tpe.suggest,
    max_evals = 1,
    trials = trials)
    
    optimal_param={}
    #Best is a dictionary that contains the indices of the optimal parameter values.
    #The following for loop uses these indices to obtain the parameter values, these values are stored in a
    #dictionary - optimal_param
    for key in best:
        optimal_param[key] = parameter_multi_layer_perceptron[key][best[key]]
        
    optimal_param['optimizer'] = parameter_multi_layer_perceptron['optimizer']
    
    #train_X = train_X.as_matrix()
    #train_Y = train_Y.as_matrix()
    train_X = StandardScaler().fit_transform(train_X)

    model = Sequential()
        
    for i in range(no_of_layers):
            
        if(i==0):
                
                model.add(Dense(output_dim = optimal_param['dim_layer'+str(i)],                                 input_dim = train_X.shape[1],                                 init = optimal_param['init_layer'+str(i)], 
                                activation = optimal_param['activation_layer'+str(i)], \
                                weights = optimal_param['weights'+str(i)], \
                                W_regularizer = optimal_param['W_regularizer'+str(i)], \
                                b_regularizer = optimal_param['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                optimal_param['activity_regularizer'+str(i)], \
                                W_constraint = optimal_param['W_constraint'+str(i)], \
                                b_constraint = optimal_param['b_constraint'+str(i)], \
                                bias = optimal_param['bias'+str(i)]))
                
                if(optimal_param['dropout'+str(i)] != 0):
                    model.add(Dropout(optimal_param['dropout'+str(i)]))
                    
        elif(i==no_of_layers-1):
            
                model.add(Dense(output_dim = 1,                                 input_dim = train_X.shape[1],                                 init = optimal_param['init_layer'+str(i)], 
                                activation = optimal_param['activation_layer'+str(i)], \
                                weights = optimal_param['weights'+str(i)], \
                                W_regularizer = optimal_param['W_regularizer'+str(i)], \
                                b_regularizer = optimal_param['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                optimal_param['activity_regularizer'+str(i)], \
                                W_constraint = optimal_param['W_constraint'+str(i)], \
                                b_constraint = optimal_param['b_constraint'+str(i)], \
                                bias = optimal_param['bias'+str(i)]))
                
        else:
            
                model.add(Dense(output_dim = optimal_param['dim_layer'+str(i)],                                 input_dim = train_X.shape[1],                                 init = optimal_param['init_layer'+str(i)], 
                                activation = optimal_param['activation_layer'+str(i)], \
                                weights = optimal_param['weights'+str(i)], \
                                W_regularizer = optimal_param['W_regularizer'+str(i)], \
                                b_regularizer = optimal_param['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                optimal_param['activity_regularizer'+str(i)], \
                                W_constraint = optimal_param['W_constraint'+str(i)], \
                                b_constraint = optimal_param['b_constraint'+str(i)], \
                                bias = optimal_param['bias'+str(i)]))
                
                if(optimal_param['dropout'+str(i)] != 0):
                    model.add(Dropout(optimal_param['dropout'+str(i)]))
                    
    model.compile(optimizer = optimal_param['optimizer'],loss = 'binary_crossentropy',metrics = ['accuracy'])
    model.fit(train_X, train_Y, nb_epoch = 10, batch_size = 128, verbose=1)

    x = model.predict_proba(train_X)
    

    return model


# In[25]:

#This function calculates the loss for different parameter values and is used to determine the most optimum 
#parameter values
def objective_multi_layer_perceptron(space_multi_layer_perceptron, data_X, data_Y, parameter_multi_layer_perceptron):
    
    global no_of_layers
    
    #Multi Layer Perceptron
    param = {}
    param['optimizer'] = parameter_multi_layer_perceptron['optimizer']
    
    for i in range(no_of_layers):
        
        param['dim_layer'+str(i)] = space_multi_layer_perceptron['dim_layer'+str(i)]
        param['init_layer'+str(i)] = space_multi_layer_perceptron['init_layer'+str(i)]
        param['activation_layer'+str(i)] = space_multi_layer_perceptron['activation_layer'+str(i)]
        param['dropout'+str(i)] = space_multi_layer_perceptron['dropout'+str(i)]
        param['weights'+str(i)] = space_multi_layer_perceptron['weights'+str(i)]
        param['W_regularizer'+str(i)] = space_multi_layer_perceptron['W_regularizer'+str(i)]
        param['b_regularizer'+str(i)] = space_multi_layer_perceptron['b_regularizer'+str(i)]
        param['activity_regularizer'+str(i)] = space_multi_layer_perceptron['activity_regularizer'+str(i)]
        param['W_constraint'+str(i)] = space_multi_layer_perceptron['W_constraint'+str(i)]
        param['b_constraint'+str(i)] = space_multi_layer_perceptron['b_constraint'+str(i)]
        param['bias'+str(i)] = space_multi_layer_perceptron['bias'+str(i)]

    
    metric_list = list()

    #Performing cross validation.
    skf=StratifiedKFold(data_Y, n_folds=3,random_state=0)
    for train_index, cross_val_index in skf:
        
        mlp_train_X, mlp_cross_val_X = data_X.iloc[train_index],data_X.iloc[cross_val_index]
        mlp_train_Y, mlp_cross_val_Y = data_Y.iloc[train_index],data_Y.iloc[cross_val_index]
        #mlp_train_X = mlp_train_X.as_matrix()
        #mlp_train_Y = mlp_train_Y.as_matrix()
        #mlp_cross_val_X = mlp_cross_val_X.as_matrix()
        #mlp_cross_val_Y = mlp_cross_val_Y.as_matrix()
        mlp_train_X = StandardScaler().fit_transform(mlp_train_X)
        mlp_cross_val_X = StandardScaler().fit_transform(mlp_cross_val_X)
        
        model = Sequential()
        
        for i in range(no_of_layers):
            
            if(i == 0):
                
                model.add(Dense(output_dim = param['dim_layer'+str(i)],                                 input_dim = mlp_train_X.shape[1],                                 init = param['init_layer'+str(i)], 
                                activation = param['activation_layer'+str(i)], \
                                weights = param['weights'+str(i)], \
                                W_regularizer = param['W_regularizer'+str(i)], \
                                b_regularizer = param['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                param['activity_regularizer'+str(i)], \
                                W_constraint = param['W_constraint'+str(i)], \
                                b_constraint = param['b_constraint'+str(i)], \
                                bias = param['bias'+str(i)]))
                
                if(param['dropout'+str(i)] != 0):
                    model.add(Dropout(param['dropout'+str(i)]))
                    
            elif(i == no_of_layers-1):
                model.add(Dense(output_dim = 1,                                 input_dim = mlp_train_X.shape[1],                                 init = param['init_layer'+str(i)], 
                                activation = param['activation_layer'+str(i)], \
                                weights = param['weights'+str(i)], \
                                W_regularizer = param['W_regularizer'+str(i)], \
                                b_regularizer = param['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                param['activity_regularizer'+str(i)], \
                                W_constraint = param['W_constraint'+str(i)], \
                                b_constraint = param['b_constraint'+str(i)], \
                                bias = param['bias'+str(i)]))
                
            else:
                
                model.add(Dense(output_dim = param['dim_layer'+str(i)],                                 input_dim = mlp_train_X.shape[1],                                 init = param['init_layer'+str(i)], 
                                activation = param['activation_layer'+str(i)], \
                                weights = param['weights'+str(i)], \
                                W_regularizer = param['W_regularizer'+str(i)], \
                                b_regularizer = param['b_regularizer'+str(i)], \
                                activity_regularizer = \
                                param['activity_regularizer'+str(i)], \
                                W_constraint = param['W_constraint'+str(i)], \
                                b_constraint = param['b_constraint'+str(i)], \
                                bias = param['bias'+str(i)]))
                
                if(param['dropout'+str(i)] != 0):
                    model.add(Dropout(param['dropout'+str(i)]))
                    
        model.compile(optimizer = param['optimizer'],loss = 'binary_crossentropy',metrics = ['accuracy'])
        model.fit(mlp_train_X, mlp_train_Y, nb_epoch = 10, batch_size = 128, verbose=1)
                    
        #score = model.evaluate(mlp_cross_val_X, mlp_cross_val_Y, verbose=0)
        #acc_list.append(score[1])
        
        predicted_values = model.predict(mlp_cross_val_X)
        
        if(metric_grid_search in ['f1', 'log_loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error', 'r2']):
            
            predictions = predicted_values >= 0.5
            predictions.astype(int)
            metric_list.append(metric_score(mlp_cross_val_Y,predictions))
            
        else :
            
            metric_list.append(metric_score(mlp_cross_val_Y,predicted_values))
            
        
    #Calculating the AUC and returning the loss, which will be minimised by selecting the optimum parameters.
    metric = np.mean(metric_list)
    return{'loss':1-metric, 'status': STATUS_OK }


# In[26]:

#Assigning the values of the multi layer perceptron parameters that need to be checked, 
#for minimizing the objective (loss). 
#The values that give the most optimum results will be picked to train the model.
def assign_space_multi_layer_perceptron(parameter_multi_layer_perceptron):
    
    global no_of_layers
    
    space_multi_layer_perceptron = {}
    #space_multi_layer_perceptron['optimizer'] = \
    #hp.choice('optimizer', parameter_multi_layer_perceptron['optimizer'])
    
    for i in range(no_of_layers):
        
        space_multi_layer_perceptron['dim_layer'+str(i)] =         hp.choice('dim_layer'+str(i), parameter_multi_layer_perceptron['dim_layer'+str(i)])
        
        space_multi_layer_perceptron['init_layer'+str(i)] =         hp.choice('init_layer'+str(i), parameter_multi_layer_perceptron['init_layer'+str(i)])
        
        space_multi_layer_perceptron['activation_layer'+str(i)] =         hp.choice('activation_layer'+str(i), parameter_multi_layer_perceptron['activation_layer'+str(i)])
        
        space_multi_layer_perceptron['dropout'+str(i)] =         hp.choice('dropout'+str(i), parameter_multi_layer_perceptron['dropout'+str(i)])
        
        space_multi_layer_perceptron['weights'+str(i)] =         hp.choice('weights'+str(i), parameter_multi_layer_perceptron['weights'+str(i)])
        
        space_multi_layer_perceptron['W_regularizer'+str(i)] =         hp.choice('W_regularizer'+str(i), parameter_multi_layer_perceptron['W_regularizer'+str(i)])
        
        space_multi_layer_perceptron['b_regularizer'+str(i)] =         hp.choice('b_regularizer'+str(i), parameter_multi_layer_perceptron['b_regularizer'+str(i)])
        
        space_multi_layer_perceptron['activity_regularizer'+str(i)] =         hp.choice('activity_regularizer'+str(i), parameter_multi_layer_perceptron['activity_regularizer'+str(i)])
        
        space_multi_layer_perceptron['W_constraint'+str(i)] =         hp.choice('W_constraint'+str(i), parameter_multi_layer_perceptron['W_constraint'+str(i)])
        
        space_multi_layer_perceptron['b_constraint'+str(i)] =         hp.choice('b_constraint'+str(i), parameter_multi_layer_perceptron['b_constraint'+str(i)])
        
        space_multi_layer_perceptron['bias'+str(i)] =         hp.choice('bias'+str(i), parameter_multi_layer_perceptron['bias'+str(i)])
        
        
    return space_multi_layer_perceptron


# In[27]:

def predict_multi_layer_perceptron(data_X, data_Y, multi_layer_perceptron):
    
    data_X = StandardScaler().fit_transform(data_X)
    #data_X = data_X.as_matrix()
    #data_Y = data_Y.as_matrix()
    
    predicted_values = multi_layer_perceptron.predict_proba(data_X)
            
    metric = metric_score(data_Y,predicted_values)

    return [metric,predicted_values]


# # Decision Tree

# In[28]:

#Trains the Decision Tree model. Performing a grid search to select the optimal parameter values
def train_decision_tree(train_X, train_Y, parameters_decision_tree):
    
    decision_tree_model = DecisionTreeClassifier()      
    model_gs = grid_search.GridSearchCV(decision_tree_model, parameters_decision_tree, scoring = metric_grid_search)
    model_gs.fit(train_X,train_Y)
    return model_gs


# In[29]:

#Predicts the output on a set of data, the built model is passed as a parameter, which is used to predict
def predict_decision_tree(data_X, data_Y, decision_tree):
    
    predicted_values = decision_tree.predict_proba(data_X)[:, 1]
    
    if(metric_grid_search in ['f1', 'log_loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error', 'r2']):
        
        predictions = decision_tree.predict(data_X)
        metric = metric_score(data_Y, predictions)
        
    else :
        
        metric = metric_score(data_Y, predicted_values)
    
    return [metric,predicted_values]


# In[30]:

def parameter_set_decision_tree(criterion = ['gini'], splitter = ['best'], max_depth = [None],                                min_samples_split = [2], min_samples_leaf = [1], min_weight_fraction_leaf = [0.0],                                max_features = [None], random_state = [None], max_leaf_nodes = [None],                                class_weight = [None], presort = [False]):
    
    parameters_decision_tree = {}
    parameters_decision_tree['criterion'] = criterion
    parameters_decision_tree['splitter'] = splitter
    parameters_decision_tree['max_depth'] = max_depth
    parameters_decision_tree['min_samples_split'] = min_samples_split
    parameters_decision_tree['min_samples_leaf'] = min_samples_leaf
    parameters_decision_tree['min_weight_fraction_leaf'] = min_weight_fraction_leaf
    parameters_decision_tree['max_features'] = max_features
    parameters_decision_tree['random_state'] = random_state
    parameters_decision_tree['max_leaf_nodes'] = max_leaf_nodes
    parameters_decision_tree['class_weight'] = class_weight
    parameters_decision_tree['presort'] = presort
    
    return parameters_decision_tree


# # Random Forest

# In[31]:

#Trains the Random Forest model. Performing a grid search to select the optimal parameter values
def train_random_forest(train_X, train_Y, parameters_random_forest):
    
    random_forest_model = RandomForestClassifier()
    model_gs = grid_search.GridSearchCV(random_forest_model, parameters_random_forest, scoring = metric_grid_search)
    model_gs.fit(train_X,train_Y)
    return model_gs


# In[32]:

#Predicts the output on a set of data, the built model is passed as a parameter, which is used to predict
def predict_random_forest(data_X, data_Y, random_forest):
    
    predicted_values = random_forest.predict_proba(data_X)[:, 1]
    
    if(metric_grid_search in ['f1', 'log_loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error', 'r2']):
        
        predictions = random_forest.predict(data_X)
        metric = metric_score(data_Y, predictions)
        
    else :
        
        metric = metric_score(data_Y, predicted_values)
    
    return [metric,predicted_values]


# In[33]:

#Parameters for random forest. To perform hyper parameter optimisation a list of multiple elements can be entered
#and the optimal value in that list will be picked using grid search
def parameter_set_random_forest(n_estimators = [10], criterion = ['gini'], max_depth = [None],                                min_samples_split = [2], min_samples_leaf = [1], min_weight_fraction_leaf = [0.0],                                max_features = ['auto'], max_leaf_nodes = [None], bootstrap = [True],                                oob_score = [False], random_state = [None], verbose = [0],warm_start = [False],                                class_weight = [None]):
    
    parameters_random_forest = {}
    parameters_random_forest['criterion'] = criterion
    parameters_random_forest['n_estimators'] = n_estimators
    parameters_random_forest['max_depth'] = max_depth
    parameters_random_forest['min_samples_split'] = min_samples_split
    parameters_random_forest['min_samples_leaf'] = min_samples_leaf
    parameters_random_forest['min_weight_fraction_leaf'] = min_weight_fraction_leaf
    parameters_random_forest['max_features'] = max_features
    parameters_random_forest['random_state'] = random_state
    parameters_random_forest['max_leaf_nodes'] = max_leaf_nodes
    parameters_random_forest['class_weight'] = class_weight
    parameters_random_forest['bootstrap'] = bootstrap
    parameters_random_forest['oob_score'] = oob_score
    parameters_random_forest['warm_start'] = warm_start
    
    return parameters_random_forest


# # Linear Regression

# In[34]:

#Trains the Linear Regression model. Performing a grid search to select the optimal parameter values
def train_linear_regression(train_X, train_Y, parameters_linear_regression):
    
    linear_regression_model = linear_model.LinearRegression()
    train_X=StandardScaler().fit_transform(train_X)
    model_gs = grid_search.GridSearchCV(linear_regression_model, parameters_linear_regression,                                        scoring = metric_grid_search)
    model_gs.fit(train_X,train_Y)
    return model_gs


# In[35]:

#Predicts the output on a set of data, the built model is passed as a parameter, which is used to predict
def predict_linear_regression(data_X, data_Y, linear_regression):
    
    data_X = StandardScaler().fit_transform(data_X)
    predicted_values = linear_regression.predict(data_X)
    
    if(metric_grid_search in ['f1', 'log_loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error', 'r2']):
        
        predictions = predicted_values >= 0.5
        predictions.astype(int)
        metric = metric_score(data_Y, predictions)
        
    else :
        
        metric = metric_score(data_Y, predicted_values)
    
    return [metric,predicted_values]


# In[36]:

#Parameters for linear regression. To perform hyper parameter optimisation a list of multiple elements can be entered
#and the optimal value in that list will be picked using grid search
def parameter_set_linear_regression(fit_intercept = [True], normalize = [False], copy_X = [True]):
    
    parameters_linear_regression = {}
    parameters_linear_regression['fit_intercept'] = fit_intercept
    parameters_linear_regression['normalize'] = normalize
    
    return parameters_linear_regression


# # Logistic Regression

# In[37]:

#Trains the Logistic Regression  model. Performing a grid search to select the optimal parameter values
def train_logistic_regression(train_X, train_Y, parameters_logistic_regression):

    logistic_regression_model = linear_model.LogisticRegression()
    train_X=StandardScaler().fit_transform(train_X)
    model_gs = grid_search.GridSearchCV(logistic_regression_model, parameters_logistic_regression,                                        scoring = metric_grid_search)
    model_gs.fit(train_X,train_Y)
    return model_gs


# In[38]:

#Predicts the output on a set of data, the built model is passed as a parameter, which is used to predict
def predict_logistic_regression(data_X, data_Y, logistic_regression):
    
    data_X = StandardScaler().fit_transform(data_X)
    predicted_values = logistic_regression.predict_proba(data_X)[:, 1]
    
    if(metric_grid_search in ['f1', 'log_loss', 'accuracy', 'mean_squared_error', 'mean_absolute_error', 'r2']):
        
        predictions = logistic_regression.predict(data_X)
        metric = metric_score(data_Y, predictions)
        
    else :
        
        metric = metric_score(data_Y, predicted_values)
    
    return [metric,predicted_values]


# In[39]:

#Parameters for logistic regression. To perform hyper parameter optimisation a list of multiple elements can be entered
#And the optimal value in that list will be picked using grid search
def parameter_set_logistic_regression(penalty = ['l2'], dual = [False], tol = [0.0001], C = [1.0],                                      fit_intercept = [True], intercept_scaling = [1], class_weight = [None],                                      random_state = [None], solver = ['liblinear'], max_iter = [100],                                      multi_class = ['ovr'], verbose = [0], warm_start = [False]):
    
    parameters_logistic_regression = {}
    parameters_logistic_regression['penalty'] = penalty
    parameters_logistic_regression['dual'] = dual
    parameters_logistic_regression['tol'] = tol
    parameters_logistic_regression['C'] = C
    parameters_logistic_regression['fit_intercept'] = fit_intercept
    parameters_logistic_regression['intercept_scaling'] = intercept_scaling
    parameters_logistic_regression['class_weight'] = class_weight
    parameters_logistic_regression['solver'] = solver
    parameters_logistic_regression['max_iter'] = max_iter
    parameters_logistic_regression['multi_class'] = multi_class
    parameters_logistic_regression['warm_start'] = warm_start
    
    return parameters_logistic_regression


# # Stacking

# In[40]:

#The stacked ensmeble will be trained by using one or more of the base model algorithms
#The function of the base model algorithm that will be used to train will be passed as the
#model_function parameter and the parameters required to train the algorithm/model will be passed as the
#model_parameters parameter
def train_stack(data_X, data_Y, model_function, model_parameters):
    
    model = model_function(data_X, data_Y, model_parameters)
    return model


# In[41]:

#Predicts the output on a set of stacked data, after the stacked model has been built by using a base model
#algorithm, hence we need the predict funcction of that base model algorithm to get the predictions
#The predict function of the base model is passed as the predict_function parameter and its respective model is 
#passed as the model parameter
def predict_stack(data_X, data_Y, predict_function, model):
    
    metric,predicted_values = predict_function(data_X, data_Y, model)
    return [metric,predicted_values]


# # Blending

# In[42]:

#The blending ensmeble will be trained by using one or more of the base model algorithms
#The function of the base model algorithm that will be used to train will be passed as the
#model_function parameter and the parameters required to train the algorithm/model will be passed as the
#model_parameters parameter
def train_blend(data_X, data_Y, model_function, model_parameters):
    
    model = model_function(blend_X, data_Y, model_parameters)
    return model


# In[43]:

#Predicts the output on a set of blended data, after the blending model has been built by using a base model
#algorithm, hence we need the predict function of that base model algorithm to get the predictions
#The predict function of the base model is passed as the predict_function parameter and its respective model is 
#passed as the model parameter
def predict_blend(data_X, data_Y, predict_function, model):
    
    metric,predicted_values = predict_function(test_blend_X, data_Y, model)
    return [metric,predicted_values]


# # Weighted Average

# In[44]:

#Perfroms weighted average of the predictions of the base models. The function that calculates the optimum 
# combination of weights is passsed as the get_weight_function parameter

#The weight_list parameter contains the weights associated with the model, they are either default weights or a list
#of weights. Using these weigths we either train the model or perform hyper parameter optimisation if there
#is a list of weights that need to be checked to find the optimum weights
def weighted_average(data_X, data_Y, hyper_parameter_optitmisation, weight_list):
    
    #Checking if hyper_parameter_optimisation is true
    if(hyper_parameter_optitmisation == True):
        
        #The last element of the weight_list which indicates wether the user wants to perform hyper parameter 
        #optimisation is deleted
        del weight_list[-1]
        #Optimisation is performed by passing the weight_list we want to optimize
        weight = get_optimized_weights(weight_list, data_X, data_Y)
    
    #Is none when performing weighted average on test data, we dont need to do anything else as we already have
    #the weights for performing the weighted average
    elif(hyper_parameter_optitmisation == None):
        
        weight = weight_list
        
    else:
       
        #The last element of the weight_list which indicates wether the user wants to perform hyper parameter 
        #optimisation is deleted
        del weight_list[-1]
        #The weight_list is now used to calculate the weighted average
        weight = weight_list
        
    weighted_avg_predictions=np.average(data_X, axis = 1, weights = weight)
    metric = metric_score(data_Y, weighted_avg_predictions)
    return [metric,weighted_avg_predictions,weight]  


# In[45]:

#Function that finds the best possible combination of weights for performing the weighted predictions.
def get_optimized_weights(weight_list, X, Y):
    
    space = assign_space_weighted_average(weight_list)
    trials = Trials()
    
    best = fmin(fn = partial(objective_weighted_average, data_X = X, data_Y = Y),
    space = space,
    algo = tpe.suggest,
    max_evals = 50,
    trials = trials)
    best_weights = list()
    
    #Arranging the weights in order of the respective models, and then returning the list of weights.
    for key in sorted(best):
        best_weights.append(best[key])
    
    return best_weights


# In[46]:

#Defining the objective. Appropriate weights need to be calculated to minimize the loss.
def objective_weighted_average(space, data_X, data_Y):
    
    weight_search_space = list()
    
    #Picking weights in the seacrh space to compute the best combination of weights
    for weight in sorted(space):
        weight_search_space.append(space[weight])
    
    weighted_avg_predictions = np.average(data_X, axis = 1, weights = weight_search_space)

    metric = metric_score(data_Y, weighted_avg_predictions)
    return{'loss':1-metric, 'status': STATUS_OK }


# In[47]:

#Assigning the weights that need to be checked, for minimizing the objective (Loss)
def assign_space_weighted_average(weight_list):
    
    space = {}
    space_index = 0
    
    for weight in weight_list:
        
        #Assigning the search space, the search space is the range of weights that need to be searched for each 
        #base model, to find the weight of that base models predictions
        space['w'+str(space_index )] = hp.choice('w'+str(space_index ), weight) 
        space_index = space_index + 1
        
    return space


# In[48]:

#The user can either use the default weights or provide their own list of values.
def assign_weights(weights = 'default',hyper_parameter_optimisation = False):
    
    weight_list = list()
    
    #The last element of the weight_list will indicate wether hyper parameter optimisation needs to be peroformed
    if(hyper_parameter_optimisation == True):
        
        if(weights == 'default'):
            
            weight_list = [range(10)] * no_of_base_models
            weight_list.append(True)
            
        else:
            
            weight_list = weights
            weight_list.append(True)
            
    else:
        
        if(weights == 'default'):
            
            weight_list = [1] * no_of_base_models
            weight_list.append(False)
            
        else:
            
            weight_list = weights
            weight_list.append(False)
    
    return weight_list


# # Setup for training and computing predictions for the models

# In[49]:

#Constructing a list (train_model_list) that contains a tuple for each base model, the tuple contains the name of 
#the function that trains the base model, and the paramters for training the base model. 

#Constructing a list (predict_model_list) that contains a tuple for each base model, the tuple contains the name of 
#the function that computes the predictions for the base model.

#In the list computed for stacking and blending, the tuples have an additional element which is the train_stack 
#function or the train_blend function. This is done because different set of data (predictions of base models) 
#needs to be passed to the base model algorithms. These function enable performing the above procedure

#These lists are constructed in such a way to enable the ease of use of the joblib library, i.e the parallel 
#module/function

def construct_model_parameter_list(model_list, parameters_list, stack = False, blend = False):
    
    model_functions = {'gradient_boosting' : [train_gradient_boosting,predict_gradient_boosting],
                       'multi_layer_perceptron' : [train_multi_layer_perceptron,predict_multi_layer_perceptron],
                       'decision_tree' : [train_decision_tree,predict_decision_tree],
                       'random_forest' : [train_random_forest,predict_random_forest],
                       'linear_regression' : [train_linear_regression,predict_linear_regression],
                       'logistic_regression' : [train_logistic_regression,predict_logistic_regression]
                      }
    
    train_model_list = list()
    predict_model_list = list()
    model_parameter_index = 0
    
    for model in model_list:
        
        if(stack == True):
            
            train_model_list.append((model_functions[model][0],parameters_list[model_parameter_index]                                         ,train_stack))
            predict_model_list.append((model_functions[model][1],predict_stack))
            
        elif(blend == True):
            
            train_model_list.append((model_functions[model][0],parameters_list[model_parameter_index]                                         ,train_blend))
            predict_model_list.append((model_functions[model][1],predict_blend))
            
        else:
            
            train_model_list.append((model_functions[model][0],parameters_list[model_parameter_index]))
            predict_model_list.append(model_functions[model][1])
            
        model_parameter_index = model_parameter_index + 1
        
    return [train_model_list,predict_model_list]


# In[50]:

#This function computes a list where each element is a tuple that contains the predict function of the base model
#along with the corresponding base model object. This is done so that the base model object can be passed to the
#predict function as a prameter to compute the predictions when using joblib's parallel module/function. 
def construct_model_predict_function_list(model_list, models,predict_model_list):
    
    model_index = 0
    model_function_list = list()
    for model in model_list:
        
        model_function_list.append((predict_model_list[model_index],models[model_index]))
        model_index = model_index + 1
    return model_function_list


# # Training base models

# In[51]:

#This function calls the respective training and predic functions of the base models.
def train_base_models(model_list, parameters_list, save_models = False):
    
    print('\nTRAINING BASE MODELS\n')
    
    #Cross Validation using Stratified K Fold
    train, cross_val = train_test_split(Data, test_size = 0.5, stratify = Data[target_label],random_state = 0)
    
    #Training the base models, and calculating AUC on the cross validation data.
    #Selecting the data (Traing Data & Cross Validation Data)
    train_Y = train[target_label]
    train_X = train.drop([target_label],axis=1)
 
    cross_val_Y = cross_val[target_label]
    cross_val_X = cross_val.drop([target_label],axis=1)
    
    #The list of base models the user wants to train.
    global base_model_list
    base_model_list = model_list

    
    #No of base models that user wants to train
    global no_of_base_models
    no_of_base_models = len(base_model_list)
    
    
    #We get the list of base model training functions and predict functions. The elements of the two lists are  
    #tuples that have (base model training function,model parameters), (base model predict functions) respectively
    [train_base_model_list,predict_base_model_list] = construct_model_parameter_list(base_model_list,                                                                                     parameters_list)
    

    #Training the base models parallely, the resulting models are stored which will be used for cross validation.
    models = (Parallel(n_jobs = -1)(delayed(function)(train_X, train_Y, model_parameter)                                                   for function, model_parameter in train_base_model_list))
    
    if(save_models == True):
        
        save_base_models(models)
    
    
    #A list with elements as tuples containing (base model predict function, and its respective model object) is 
    #returned. This list is used in the next step in the predict_base_models function, the list will be used in
    #joblibs parallel module/function to compute the predictions and metric scores of the base models
    #Appended in the following manner so it can be used in joblib's parallel module/function
    global base_model_predict_function_list
    base_model_predict_function_list = construct_model_predict_function_list(base_model_list, models,                                                                        predict_base_model_list)
    predict_base_models(cross_val_X, cross_val_Y,mode = 'train')


# # Predictions of base models

# In[52]:

def predict_base_models(data_X, data_Y,mode):
    
    print('\nTESTING/CROSS VALIDATION BASE MODELS\n')
    
    predict_list = list()
    
    predict_gradient_boosting = list()
    predict_multi_layer_perceptron = list()
    predict_decision_tree = list()
    predict_random_forest = list()
    predict_linear_regression = list()
    predict_logistic_regression = list()
    
    metric_linear_regression = list()
    metric_logistic_regression = list()
    metric_decision_tree = list()
    metric_random_forest = list()
    metric_multi_layer_perceptron = list()
    metric_gradient_boosting = list()
    
    auc_predict_index = 0
    
    #Initializing a list which will contain the predictions of the base models and the variables that will
    #calculate the metric score
    model_predict_metric = {'gradient_boosting' : [predict_gradient_boosting, metric_gradient_boosting],
                       'multi_layer_perceptron' : [predict_multi_layer_perceptron, metric_multi_layer_perceptron],
                       'decision_tree' : [predict_decision_tree, metric_decision_tree],
                       'random_forest' : [predict_random_forest, metric_random_forest],
                       'linear_regression' : [predict_linear_regression, metric_linear_regression],
                       'logistic_regression' : [predict_logistic_regression, metric_logistic_regression]
                      }
    
    #Computing the AUC and Predictions of all the base models on the cross validation data parallely.
    auc_predict_cross_val = (Parallel(n_jobs = -1)(delayed(function)(data_X, data_Y, model)
                                               for function, model in base_model_predict_function_list))
    
    #Building the list which will contain all the predictions of the base models and will also display the metric
    #scores of the base models
    for model in base_model_list:
        
        #Assigning the predictions and metrics computed for the respective base model
        model_predict_metric[model] = auc_predict_cross_val[auc_predict_index][1],        auc_predict_cross_val[auc_predict_index][0]
        auc_predict_index = auc_predict_index + 1
        
        if(model == 'multi_layer_perceptron'):
            
            #This is done only for multi layer perceptron because the predictions returned by the multi layer 
            #perceptron model is a list of list, the below piece of code converts this nested list into a single
            #list
            predict_list.append(np.asarray(sum(model_predict_metric[model][0].tolist(), [])))
            
        else:
            
            #The below list will contain all the predictions of the base models.
            predict_list.append(model_predict_metric[model][0])
        
        #Printing the name of the base model and its corresponding metric score
        print_metric(model,model_predict_metric[model][1])
    
    if(mode == 'train'):
        
        #Function to construct dataframes for training the second level/ensmeble models using the predictions of the
        #base models on the train dataset
        second_level_train_data(predict_list, data_X, data_Y)
        
    else:
        
        #Function to construct dataframes for testing the second level/ensmeble models using the predictions of the
        #base models on the test dataset
        second_level_test_data(predict_list, data_X, data_Y)


# # Saving  models

# In[53]:

#The trained base model objects can be saved and used later for any other purpose. The models asre save using 
#joblib's dump. The models are named base_model1, base_model2..so on depending on the order entered by the user
#while training these models in the train_base_model function
def save_base_models(models):
    
    model_index = 0
    
    for model in models:
        
        joblib.dump(model, 'base_model'+str(model_index)+'.pkl')
        model_index = model_index + 1


# In[54]:

#This function will return the trained base model objects once they have been saved in the function above. All the 
#trained models are returned in a list called models
def get_base_models():
    
    models = list()
    
    for model_index in range(no_of_base_models):
        models.append(joblib.load('base_model'+str(model_index)+'.pkl'))
        
    return models


# # Training the ensemble/second level models

# In[55]:

#Training the second level models parallely
def train_ensemble_models(stack_model_list = [], stack_parameters_list = [], blend_model_list = [],                              blend_parameters_list = [], perform_weighted_average = False, weights_list = None,
                          save_models = False):
    
    print('\nTRAINING ENSEMBLE MODELS\n')
    
    global no_of_ensemble_models
    
    #This list will contain the names of the models/algorithms that have been used as second level models
    #This list will be used later in the testing phase for identifying which model belongs to which ensemble
    #(stacking or blending), hence the use of dictionaries as elements of the list
    #Analogous to the base_model_list
    global ensmeble_model_list
    ensmeble_model_list = list()
    
    train_stack_model_list = list() 
    predict_stack_model_list = list()
    train_blend_model_list = list()
    predict_blend_model_list = list()
    
    #The list will be used to train the ensemble models, while using joblib's parallel
    train_second_level_models = list() 
    
    #Stacking will not be done if user does not enter the list of models he wants to use for stacking
    if(stack_model_list != []):
        
        #Appending a dictionary that contians key-Stacking and its values/elements are the names of the 
        #models/algorithms that are used for performing the stacking procedure, this is done so that it will be easy
        #to identify the models belonging to the stacking ensemble
        ensmeble_model_list.append({'Stacking' : stack_model_list})
        
        #We get the list of stacked model training functions and predict functions. The elements of the two   
        #lists are tuples that have(base model training function,model parameters,train_stack function),
        #(base model predict functions,predict_stack function) respectively
        [train_stack_model_list,predict_stack_model_list] = construct_model_parameter_list(stack_model_list,                                                                                           stack_parameters_list,
                                                                                           stack=True)
        
    #Blending will not be done if user does not enter the list of models he wants to use for blending
    if(blend_model_list != []):
        
        #Appending a dictionary that contians key-Blending and its values/elements are the names of the 
        #models/algorithms that are used for performing the blending procedure, this is done so that it will be easy
        #to identify the models belonging to the blending ensemble
        ensmeble_model_list.append({'Blending' : blend_model_list})

        #We get the list of blending model training functions and predict functions. The elements of the two   
        #lists are tuples that have(base model training function,model parameters,train_blend function),
        #(base model predict functions,predict_blend function) respectively
        [train_blend_model_list,predict_blend_model_list] = construct_model_parameter_list(blend_model_list,                                                                                           blend_parameters_list,                                                                                           blend=True)
        
    #The new list contains either the stacked models or blending models or both or remain empty depending on what 
    #the user has decided to use
    train_second_level_models = train_stack_model_list + train_blend_model_list
    
    #If the user wants to perform a weighted average, a tuple containing (hyper parmeter optimisation = True/False,
    #the lsit of weights either deafult or entered by the user, and the function that performs the weighted average)
    #will be created. This tuple will be appended to the list above
    #weights_list[-1] is an element of the list that indicates wwether hyper parameter optimisation needs to be
    #perofrmed
    if(perform_weighted_average == True):
        
        train_weighted_average_list = (weights_list[-1], weights_list, weighted_average)
        train_second_level_models.append(train_weighted_average_list)

        
    no_of_ensemble_models = len(train_second_level_models)

    #If weighted average is performed, the last element of models will contain the metric score and weighted average
    #predictions, and not a model object. So we use the last element in different ways compared to the other model
    #objects
    
    #Training the ensmeble models parallely 
    models = Parallel(n_jobs = -1)(delayed(function)(stack_X, stack_Y, model, model_parameter)                                        for model, model_parameter, function in train_second_level_models)
    
    
    #A list with elements as tuples containing((base model predict function,predict_stack or predict_blend functions)
    #,and its respective base model object) is returned. This list is used in the next step in the   
    #predict_ensemble_models function, the list will be used in
    #joblibs parallel module/function to compute the predictions and metric score of the ensemble models
    #Appended in the following manner so it can be used in joblib's parallel module/function
    #Analogous to base_model_predict_function_list
    global ensmeble_model_predict_function_list
    ensmeble_model_predict_function_list = construct_model_predict_function_list(stack_model_list + blend_model_list,                                                                                 models, predict_stack_model_list 
                                                                                 + predict_blend_model_list)
    
    #If weighted average is needed to be perofrmed we need to append((None(which indicates its testing phase),the
    #weighted average function),and the weights). Appended in the following manner so it can be used in joblib's
    #parallel module/function
    if(perform_weighted_average == True):
        
        weight = models[-1][-1]
        print('Weighted Average')
        print('Weight',weight)
        print('Metric Score',models[-1][0])
        ensmeble_model_list.append({'Weighted Average' : [str(weight)]})
        ensmeble_model_predict_function_list.append(((None,weighted_average),weight))
        
    if(save_models == True and perform_weighted_average == True):
        
        del models[-1]
        no_of_ensemble_models = no_of_ensemble_models - 1
        save_ensemble_models(models)
        
    elif(save_models == True and perform_weighted_average == False):
        
        save_ensemble_models(models)


# # Prediction of ensemble models

# In[56]:

def predict_ensemble_models(data_X, data_Y):
    
    print('\nTESTING ENSEMBLE MODELS\n')

    metric_linear_regression = list()
    metric_logistic_regression = list()
    metric_decision_tree = list()
    metric_random_forest = list()
    metric_multi_layer_perceptron = list()
    metric_gradient_boosting = list()
    metric_weighted_average = list()
    metric_stacking = list()
    metric_blending = list()
    
    auc_predict_index = 0
    
    #Initializing a list which will contain the predictions of the base models and the variables that will
    #calculate the metric score
    model_metric = {'gradient_boosting' : [metric_gradient_boosting],
                       'multi_layer_perceptron' : [metric_multi_layer_perceptron],
                       'decision_tree' : [metric_decision_tree],
                       'random_forest' : [metric_random_forest],
                       'linear_regression' : [metric_linear_regression],
                       'logistic_regression' : [metric_logistic_regression]
                      }
    
    #Computing the AUC and Predictions of all the ensmeble models on the test data parallely.
    auc_predict_cross_val = (Parallel(n_jobs = -1)(delayed(function[1])(data_X, data_Y, function[0],model)
                                               for function, model in ensmeble_model_predict_function_list))
    
    #ensemble_model_list is a list defined in the train_ensemble_models function, each element of the lsit is a
    #dictionary, that contains the name of the ensembling technique (key) and the models assocaited with it(values)
    
    #So the first for loop gives the dictionary
    for ensemble_models in ensmeble_model_list:
    
        #This for gives the key value pair, key being the name of the ensembling technique, value being a list
        #of the models used for that ensemble
        for ensemble,models in ensemble_models.items():
            
            #This for loop gives the iterates through the models present in the models list adn asssigns 
            #the metric score and prints it
            for model in models:
                
                #Assigning the predictions and metrics computed for the respective ensmeble model
                model_metric[model] = auc_predict_cross_val[auc_predict_index][0]
                auc_predict_index = auc_predict_index + 1
        
                #Printing the name of the ensmeble technique and its model and its corresponding metric score
                print_metric(ensemble + " " + model,model_metric[model])


# In[57]:

#The trained ensmeble model objects can be saved and used later for any other purpose. The models asre save using 
#joblib's dump. The models are named ensmeble_model1, emnsmeble_model2..so on depending on the order entered by 
#the user while training these models in the train_base_model function
def save_ensemble_models(models):
    
    model_index = 0
    
    for model in models:
        
        joblib.dump(model, 'ensemble_model'+str(model_index)+'.pkl')
        model_index = model_index + 1


# In[58]:

#This function will return the trained base model objects once they have been saved in the function above. All the 
#trained models are returned in a list called models
def get_ensemble_models():
    
    models = list()
    
    for model_index in range(no_of_ensemble_models):
        models.append(joblib.load('ensemble_model'+str(model_index)+'.pkl'))
        
    return models


# In[59]:

def test_models(test_data):
    
    print('\nTESTING PHASE\n')
    
    #Training the base models, and calculating AUC on the test data.
    #Selecting the data (Test Data)
    test_Y = test_data[target_label]
    test_X = test_data.drop([target_label],axis=1)
    
    predict_base_models(test_X,test_Y,mode='test')
    predict_ensemble_models(test_stack_X,test_stack_Y)


# In[60]:

def print_metric(model,metric_score):
 
    #Printing the metric score for the corresponding model.
    print (model,'\n',metric_score)

