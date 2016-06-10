
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing, grid_search
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, KFold
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from joblib import Parallel, delayed
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials 
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional


# In[2]:

#Reading the data, into a Data Frame.
Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)

#Encoding the data, encoding the string values into numerical values.
encode = preprocessing.LabelEncoder()

#Selcting the columns of string data type
names = Data.select_dtypes(include = ['object'])

#Function that encodes the string values to numerical values.
def enc(data,column):
    data[column] = encode.fit_transform(data[column])
    return data
for column in names:
        Data = enc(Data,column)


# # Sample Generation

# In[3]:

#Performing training, cross validation and testing on different stratified splits of the data.
def sample_generation(n):
    
    for i in range(n):
        data_initialize()
        data_split()
        metric_initialize()
        train_cross_val_base_models()
        print_metric_cross_val(i)
        train_stack_blend()
        metric_initialize()
        test_data()
        print_metric_test(i)


# In[4]:

#Splitting the data into training and testing datasets (Stratified Split)
def data_split():
    
    global Data
    global test
    Data, test = train_test_split(Data, test_size = 0.1, stratify = Data['y'])


# In[5]:

#This function is used to convert the predictions of the base models into a DataFrame.
def build_data_frame(data):
    
    data_frame = pd.DataFrame(data).T
    return data_frame


# In[6]:

def data_initialize():
    
    #Initializing the test dataset.
    test = pd.DataFrame()
    
    global stack_X
    global stack_Y
    
    #Initializing two data frames that will be used as training data for the stacked model.
    #The data frame will contain the predictions of the base models.
    stack_X = pd.DataFrame() 
    #The data frame will contain the calss labels of the base models.
    stack_Y = pd.DataFrame() 
    
    global blend_X
    global raw_features_X
    
    #Initializing two data frames that will be used as training data for the blending model.
    #The data frames will contain the predictions and raw features  of the base models.
    blend_X = pd.DataFrame() 
    #The data frames will contain the raw features  of the data, which will be concatenated with the predictions.
    raw_features_X = pd.DataFrame() 
    
    global test_blend_X
    global test_raw_features_X
    global test_stack_X 
    global test_stack_Y
    
    #Initializing the dataframes that will be used for testing the stacking and blending models.
    test_blend_X = pd.DataFrame()
    test_raw_features_X = pd.DataFrame()
    test_stack_X = pd.DataFrame()
    test_stack_Y = pd.DataFrame() 


# In[7]:

#Defining the parameters for the XGBoost (Gradient Boosting) Algorithm.
def param_set():
    
    #Gradient Boosting (XGBoost)
    param = {}
    #Setting Parameters for the Booster
    param['booster'] = ['gbtree']
    param['objective'] = ['binary:logistic']
    param["eval_metric"] = ["auc"]
    param['eta'] = [0.1,0.3,0.5,0.7,0.9]
    param['gamma'] = [0,1,5,10]
    param['max_depth'] = [6,9,12,15,18,21]
    param['min_child_weight'] = [1,5,10]
    param['max_delta_step'] = [0,1,10]
    param['subsample'] = [0.5,1]
    param['colsample_bytree'] = [0.5,1]
    param['silent'] = [1]
    param['seed'] = [0]
    param['base_score'] = [0.5]
    param['lambda_bias'] = [1,5,10]
    
    return param


# In[8]:

#Defining the parameters for the XGBoost (Gradient Boosting) Algorithm.
def param_sett():
    
    #Gradient Boosting (XGBoost)
    param = {}
    #Setting Parameters for the Booster
    param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    param["eval_metric"] = "auc"
    param['eta'] = 0.3
    param['gamma'] = 0
    param['max_depth'] = 6
    param['min_child_weight'] = 1
    param['max_delta_step'] = 0
    param['subsample'] = 1
    param['colsample_bytree'] = 1
    param['silent'] = 1
    param['seed'] = 0
    param['base_score'] = 0.5
    param['lambda_bias'] = 1
    
    return param


# In[9]:

#Assigning the weights that need to be checked, for minimizing the objective (Loss)
def assign_space_gradient_boosting():
    
    parameter_gradient_boosting = param_set()
    space_gradient_boosting ={
        
        'eta': hp.choice('eta', parameter_gradient_boosting['eta']),
        
        'gamma': hp.choice('gamma', parameter_gradient_boosting['gamma']),
        
        'max_depth': hp.choice('max_depth', parameter_gradient_boosting['max_depth']),
        
        'min_child_weight': hp.choice('min_child_weight', parameter_gradient_boosting['min_child_weight']),
        
        'max_delta_step': hp.choice('max_delta_step', parameter_gradient_boosting['max_delta_step']),
        
        'subsample': hp.choice('subsample', parameter_gradient_boosting['subsample']),
        
        'colsample_bytree': hp.choice('colsample_bytree', parameter_gradient_boosting['colsample_bytree']),
        
        'silent': hp.choice('silent', parameter_gradient_boosting['silent']),
        
        'seed': hp.choice('seed', parameter_gradient_boosting['seed']),
        
        'base_score': hp.choice('base_score', parameter_gradient_boosting['base_score']),
        
        'lambda_bias': hp.choice('lambda_bias', parameter_gradient_boosting['lambda_bias'])
        
    }
    
    return space_gradient_boosting


# In[10]:

#Trains the Gradient Boosting model.
def train_gradient_boosting(train_X,train_Y):
    
    model = gradient_boosting_parameters(train_X,train_Y)
    return model
    


# In[11]:

def gradient_boosting_parameters(train_X,train_Y):
    
    space_gradient_boosting = assign_space_gradient_boosting()
    trials = Trials()
    
    best = fmin(fn = objective_gradient_boosting,
    space = space_gradient_boosting,
    algo = tpe.suggest,
    max_evals = 100,
    trials = trials)
    
    parameter_gradient_boosting = param_set()
    for key in best:
        best[key] = parameter_gradient_boosting[key][best[key]]
        
    best['booster'] = 'gbtree'
    best['objective'] = 'binary:logistic'
    best["eval_metric"] = "auc"
    print(best)
    dtrain = xgb.DMatrix(train_X, label = train_Y)
    model = xgb.train(best, dtrain)
    return model


# In[12]:

def objective_gradient_boosting(space_gradient_boosting):
    
    #Gradient Boosting (XGBoost)
    param = {}
    #Setting Parameters for the Booster
    param['booster'] = 'gbtree'
    param['objective'] = 'binary:logistic'
    param["eval_metric"] = "auc"
    param['eta'] = space_gradient_boosting['eta']
    param['gamma'] = space_gradient_boosting['gamma']
    param['max_depth'] = space_gradient_boosting['max_depth']
    param['min_child_weight'] = space_gradient_boosting['min_child_weight']
    param['max_delta_step'] = space_gradient_boosting['max_delta_step']
    param['subsample'] = space_gradient_boosting['subsample']
    param['colsample_bytree'] = space_gradient_boosting['colsample_bytree']
    param['silent'] = space_gradient_boosting['silent']
    param['seed'] = space_gradient_boosting['seed']
    param['base_score'] = space_gradient_boosting['base_score']
    param['lambda_bias'] = space_gradient_boosting['lambda_bias']
    
    model = xgb.Booster()
    auc_list = list()
    
    skf=StratifiedKFold(train_Y, n_folds=3)
    for train_index, cross_val_index in skf:
        
        xgb_train_X, xgb_cross_val_X = train_X.iloc[train_index],train_X.iloc[cross_val_index]
        xgb_train_Y, xgb_cross_val_Y = train_Y.iloc[train_index],train_Y.iloc[cross_val_index]
        
        dtrain = xgb.DMatrix(xgb_train_X, label = xgb_train_Y)
        model = xgb.train(param, dtrain)
        
        predict = model.predict(xgb.DMatrix(xgb_cross_val_X, label = xgb_cross_val_Y))
        auc_list.append(roc_auc_score(xgb_cross_val_Y,predict))
        
    auc = np.mean(auc_list)
    return{'loss':1-auc, 'status': STATUS_OK }


# In[13]:

#Trains the Multi Layer Perceptron model.
def train_multi_layer_perceptron(train_X,train_Y):
    
    model = Sequential()
    model.add(Dense(output_dim = 64, input_dim = train_X.shape[1], init = 'uniform', activation = 'sigmoid'))
    model.add(Dense(output_dim = 1, input_dim = 64,activation = 'sigmoid',))
    model.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])
    model.fit(train_X.as_matrix(), train_Y.as_matrix(), nb_epoch = 5, batch_size = 128)
    return model


# In[14]:

#Trains the Decision Tree model.
def train_decision_tree(train_X,train_Y):
    
    model = DecisionTreeClassifier()
    param = decision_tree_parameters({'max_depth':[6,9,12,15,20],'criterion':['gini','entropy'],})
    model_cv = grid_search.GridSearchCV(model, param)
    model_cv.fit(train_X,train_Y)
    print(model_cv.best_estimator_)
    return model_cv


# In[15]:

#Trains the Random Forest model.
def train_random_forest(train_X,train_Y):
    
    model = RandomForestClassifier()
    param = random_forest_parameters({'max_depth':[6,9,12,15,20],'n_estimators':[5,10,15,20]})
    model_cv = grid_search.GridSearchCV(model, param)
    model_cv.fit(train_X,train_Y)
    print(model_cv.best_estimator_)
    return model_cv


# In[16]:

#Trains the Linear Regression model.
def train_linear_regression(train_X,train_Y):
    
    model = linear_model.LinearRegression()
    #Scaling the data
    train_X = preprocessing.StandardScaler().fit_transform(train_X)
    param = linear_regression_parameters()
    model_cv = grid_search.GridSearchCV(model, param)
    model_cv.fit(train_X,train_Y)
    print(model_cv.best_estimator_)
    return model_cv


# In[17]:

#Trains the Logistic Regression (L2) model.
def train_logistic_regression_L1(train_X,train_Y):
    
    model = linear_model.LogisticRegression()
    #Scaling the data
    train_X = preprocessing.StandardScaler().fit_transform(train_X)
    param = logistic_regression_L1_parameters({'penalty':['l1'],'C':[0.0001,0.001,0.01,0.1,1,10,100,100]})
    model_cv = grid_search.GridSearchCV(model, param)
    model_cv.fit(train_X,train_Y)
    print(model_cv.best_estimator_)
    return model_cv


# In[18]:

#Trains the Logistic Regression (L2) model.
def train_logistic_regression_L2(train_X,train_Y):
    
    model = linear_model.LogisticRegression()
    #Scaling the data
    train_X = preprocessing.StandardScaler().fit_transform(train_X)
    param = logistic_regression_L2_parameters({'penalty':['l2'],'C':[0.0001,0.001,0.01,0.1,1,10,100,100]})
    model_cv = grid_search.GridSearchCV(model, param,cv=5)
    model_cv.fit(train_X,train_Y)
    print(model_cv.best_estimator_)
    return model_cv


# In[19]:

#Trains the Stacking model (Gradient Boosting - XGBoost)
def train_stack_model(train_X,train_Y):
    
    model = xgb.Booster()
    param = param_sett()
    dtrain = xgb.DMatrix(train_X,label = train_Y)
    model = xgb.train(param, dtrain)
    return model


# In[20]:

#Trains the blending model (Gradient Boosting - XGBoost)
def train_blend_model(train_X,train_Y): 
    
    model = xgb.Booster()
    param = param_sett()
    dtrain = xgb.DMatrix(train_X,label = train_Y)
    model = xgb.train(param, dtrain)
    return model


# In[21]:

def cross_val_gradient_boosting(cross_val_X,cross_val_Y):
    
    predict = gradient_boosting.predict(xgb.DMatrix(cross_val_X, label = cross_val_Y))
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[22]:

def cross_val_multi_layer_perceptron(cross_val_X,cross_val_Y):
    
    predict = multi_layer_perceptron.predict_on_batch(cross_val_X)
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[23]:

def cross_val_decision_tree(cross_val_X,cross_val_Y):
    
    global decision_tree
    predict = decision_tree.predict_proba(cross_val_X)[:,1]
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[24]:

def cross_val_random_forest(cross_val_X,cross_val_Y):
    
    predict = random_forest.predict_proba(cross_val_X)[:,1]
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[25]:

def cross_val_linear_regression(cross_val_X,cross_val_Y):
    
    cross_val_X = preprocessing.StandardScaler().fit_transform(cross_val_X)
    predict = linear_regression.predict(cross_val_X)
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[26]:

def cross_val_logistic_regression_L1(cross_val_X,cross_val_Y):
    
    cross_val_X = preprocessing.StandardScaler().fit_transform(cross_val_X)
    predict = logistic_regression_L1.predict_proba(cross_val_X)[:,1]
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[27]:

def cross_val_logistic_regression_L2(cross_val_X,cross_val_Y):
    
    cross_val_X = preprocessing.StandardScaler().fit_transform(cross_val_X)
    predict = logistic_regression_L2.predict_proba(cross_val_X)[:,1]
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[28]:

def cross_val_stack(cross_val_X,cross_val_Y):

    predict = stack.predict(xgb.DMatrix(cross_val_X,label = cross_val_Y))
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[29]:

def cross_val_blend(cross_val_X,cross_val_Y):

    predict = blend.predict(xgb.DMatrix(cross_val_X,label = cross_val_Y))
    auc = roc_auc_score(cross_val_Y,predict)
    return [auc,predict]


# In[30]:

#Perfroms weighted average of the predictions of the base models.
def weighted_average(data_frame_predictions, cross_val_Y,weight):
    
    weighted_avg_predictions=np.average(data_frame_predictions,axis=1,weights=weight)
    auc = roc_auc_score(cross_val_Y,weighted_avg_predictions)
    return [auc,weighted_avg_predictions]  


# In[31]:

#Defining the objective. Appropriate weights need to be calculated to minimize the loss.
def objective_weighted_average(space):
    
    global stack_X
    weighted_avg_predictions=np.average(stack_X,axis=1,weights=[space['w1'],space['w2'],space['w3']
                                                                              ,space['w4'],space['w5'],
                                                                              space['w6'],space['w7']])
    
    
    global stack_Y
    auc = roc_auc_score(stack_Y,weighted_avg_predictions)
    return{'loss':1-auc, 'status': STATUS_OK }


# In[32]:

#Assigning the weights that need to be checked, for minimizing the objective (Loss)
def assign_space_weighted_average():
    
    space ={
        'w1': hp.choice("x_w1", range(10)),
        'w2': hp.choice('x_w2', range(10)),
        'w3': hp.choice('x_w3', range(10)),
        'w4': hp.choice('x_w4', range(10)),
        'w5': hp.choice('x_w5', range(10)),
        'w6': hp.choice('x_w6', range(10)),
        'w7': hp.choice('x_w7', range(10))
    }
    
    return space


# In[33]:

#Function that finds the best possible combination of weights for performing the weighted predictions.
def get_weights():
    
    space = assign_space_weighted_average()
    trials = Trials()
    
    best = fmin(fn = objective_weighted_average,
    space = space,
    algo = tpe.suggest,
    max_evals = 100,
    trials = trials)
    best_weights = list()
    
    #Arranging the weights in order of the respective models, and then returning the list of weights.
    for key in sorted(best):
        best_weights.append(best[key])
    
    return best_weights


# In[34]:

def metric_initialize():
    
    global metric_linear_regression
    global metric_logistic_regression_L2
    global metric_logistic_regression_L1
    global metric_decision_tree
    global metric_random_forest
    global metric_gradient_boosting
    global metric_multi_layer_perceptron
    global metric_stacking
    global metric_blending
    global metric_weighted_average
    
    #Initialzing the variables that will be used to calculate the area under the curve on the given data.
    metric_linear_regression = list()
    metric_logistic_regression_L2 = list()
    metric_logistic_regression_L1 = list()
    metric_decision_tree = list()
    metric_random_forest = list()
    metric_multi_layer_perceptron = list()
    metric_gradient_boosting = list()
    metric_weighted_average = list()
    metric_stacking = list()
    metric_blending = list()


# In[35]:

def decision_tree_parameters(parameters_decision_tree={}):
    
    param = parameters_decision_tree
    return param


# In[36]:

def random_forest_parameters(parameters_random_forest={}):
    
    param = parameters_random_forest
    return param


# In[37]:

def linear_regression_parameters(parameters_linear_regression={}):
    
    param = parameters_linear_regression
    return param


# In[38]:

def logistic_regression_L1_parameters(parameters_logistic_regression_L1={}):
    
    param = parameters_logistic_regression_L1
    return param


# In[39]:

def logistic_regression_L2_parameters(parameters_logistic_regression_L2={}):
    
    param = parameters_logistic_regression_L2
    return param


# In[40]:

#The list of base model functions (Training).
train_base_model_list = [train_gradient_boosting,train_multi_layer_perceptron,train_decision_tree,train_random_forest,
                 train_linear_regression,train_logistic_regression_L1,train_logistic_regression_L2]

#The list of base model functions (Cross Validation).
cross_val_base_model_list = [cross_val_gradient_boosting,cross_val_multi_layer_perceptron,cross_val_decision_tree
                           ,cross_val_random_forest,cross_val_linear_regression,cross_val_logistic_regression_L1
                           ,cross_val_logistic_regression_L2]

#The list of second level model functions.
cross_val_second_level_model = [cross_val_stack,cross_val_blend]


# # Base Model Predictions

# In[41]:

def train_cross_val_base_models():
    
    #Cross Validation using Stratified K Fold
    train, cross_val = train_test_split(Data, test_size = 0.5, stratify = Data['y'])
    
    #Training the base models, and calculating AUC on the cross validation data.
    #Selecting the data (Traing Data & Cross Validation Data)
    global train_X
    global train_Y
    train_Y = train['y']
    train_X = train.drop(['y'],axis=1)
    global cross_val_X
    global cross_val_Y
    cross_val_Y = cross_val['y']
    cross_val_X = cross_val.drop(['y'],axis=1)
    
    global gradient_boosting
    global multi_layer_perceptron
    global decision_tree
    global random_forest
    global linear_regression
    global logistic_regression_L1
    global logistic_regression_L2

    #Training the base models parallely, the resulting models are stored which will be used for cross validation.
    [gradient_boosting,multi_layer_perceptron,decision_tree,random_forest,linear_regression,logistic_regression_L1
     ,logistic_regression_L2] = (Parallel(n_jobs = -1)(delayed(function)(train_X, train_Y)\
                                                   for function in train_base_model_list))
    
    #Computing the AUC and Predictions of all the base models on the cross validation data parallely.
    auc_predict_cross_val = (Parallel(n_jobs = -1)(delayed(function)(cross_val_X,cross_val_Y)                                               for function in cross_val_base_model_list))
    
    #Gradient Boosting (XGBoost)
    #The AUC error (Cross Validation Data)
    auc,predict_gradient_boosting = auc_predict_cross_val[0][0],auc_predict_cross_val[0][1]
    metric_gradient_boosting.append(auc)
    
    #Multi Layer Perceptron
    #The AUC (Cross Validation Data)
    predict_mlp=list()
    auc,predict_multi_layer_perceptron = auc_predict_cross_val[1][0],auc_predict_cross_val[1][1]
    metric_multi_layer_perceptron.append(auc)
    
    #predict_multi_layer_perceptron returns a list of lists containing the predictions,
    #this cannot be converted to a dataframe.
    #This inner lists are converted to floats and then used to convert it to a dataframe.
    for i in predict_multi_layer_perceptron:
        predict_mlp.append(float(i))
    
    #Decision Tree)
    #The AUC (Cross Validation Data)
    auc,predict_decision_tree = auc_predict_cross_val[2][0],auc_predict_cross_val[2][1]
    metric_decision_tree.append(auc)
    
    #Random Forest (Deafult=10 Trees)
    #The AUC (Cross Validation Data)
    auc,predict_random_forest = auc_predict_cross_val[3][0],auc_predict_cross_val[3][1]
    metric_random_forest.append(auc)
    
    #Linear Regression
    #The AUC (Cross Validation Data)
    auc,predict_linear_regression = auc_predict_cross_val[4][0],auc_predict_cross_val[4][1]
    metric_linear_regression.append(auc)
    
    #Logistic Regression (Default=l2)
    #The AUC (Cross Validation Data)
    auc,predict_logistic_regression_L1 = auc_predict_cross_val[5][0],auc_predict_cross_val[5][1]
    metric_logistic_regression_L1.append(auc)
    
    #Logistic Regression-L2
    #The AUC (Cross Validation Data)
    auc,predict_logistic_regression_L2 = auc_predict_cross_val[6][0],auc_predict_cross_val[6][1]
    metric_logistic_regression_L2.append(auc)
    
    #Building a list that contains all the predictions of the base models.
    predict_list = [predict_gradient_boosting,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1,predict_mlp]
    
    #Converting the above list of predictions into a dataframe, which will be used to train the stacking model.
    global stack_X
    stack_X = stack_X.append(build_data_frame(predict_list))
    
    #Building a list that contains all the raw features, used as cross validation data for the base models.
    global raw_features_X
    raw_features_X = raw_features_X.append(cross_val_X,ignore_index=True)
    
    #Storing the cross validation dataset labels in the variable stack_Y, 
    #which will be used later to train the stacking and blending models.
    global stack_Y
    stack_Y = cross_val_Y  
    
    #Performing a weighted average of all the base models and calculating the resulting AUC.
    global weight
    weight = get_weights()
    
    auc,predict_weighted_average = weighted_average(stack_X,stack_Y,weight)
    metric_weighted_average.append(auc)



# In[42]:

def print_metric_cross_val(n):
    
    #Calculating the average AUC across all the AUC computed on the cross validation folds.
    avg_linear_regression = np.mean(metric_linear_regression)
    avg_logistic_regression_L2 = np.mean(metric_logistic_regression_L2)
    avg_logistic_regression_L1 = np.mean(metric_logistic_regression_L1)
    avg_decision_tree = np.mean(metric_decision_tree)
    avg_random_forest = np.mean(metric_random_forest)
    avg_multi_layer_perceptron = np.mean(metric_multi_layer_perceptron)
    avg_gradient_boosting = np.mean(metric_gradient_boosting)
    
    #Printing the AUC for the base models.
    print('\nStart Cross Validation Sample',n,'\n')
    print (' AUC (Linear Regression)\n',avg_linear_regression)
    print (' AUC (Logistic Regression - L2)\n',avg_logistic_regression_L2)
    print (' AUC (Logistic Regression - L1)\n',avg_logistic_regression_L1)
    print (' AUC (Decision Tree)\n',avg_decision_tree)
    print (' AUC (Random Forest)\n',avg_random_forest)
    print (' AUC (Multi Layer Perceptron)\n',avg_multi_layer_perceptron)
    print (' AUC (Weighted Average)\n',metric_weighted_average)
    print (' AUC (Gradient Boosting - XGBoost)\n',avg_gradient_boosting)
    print('\nEnd Cross Validation Sample',n,'\n')
    


# In[43]:

def train_stack_blend():
    
    #Converting the above list of predictions and raw features (Concatenate) into a dataframe, 
    #which will be used to train the blending model.
    global blend_X
    blend_X = pd.concat([raw_features_X, stack_X], axis = 1,ignore_index = True)
    
    #Training the Stacking and Blending models parallely using the 
    #predictions of base models on the cross validation data.
    global stack
    global blend
    function_param = [(train_stack_model,stack_X,stack_Y),(train_blend_model,blend_X,stack_Y)]
    [stack,blend] = Parallel(n_jobs = -1)(delayed(model_function)(train_X,train_Y)                                        for model_function,train_X,train_Y in function_param)
    


# In[44]:

def print_metric_test(n):
    
    print('\nStart Test Sample',n,'\n')
    #Printing the AUC for all the models. (Test Data)
    print (' AUC (Linear Regression)\n',metric_linear_regression)
    print (' AUC (Logistic Regression - L2)\n',metric_logistic_regression_L2)
    print (' AUC (Logistic Regression - L1)\n',metric_logistic_regression_L1)
    print (' AUC (Decision Tree)\n',metric_decision_tree)
    print (' AUC (Random Forest)\n',metric_random_forest)
    print (' AUC (Multi Layer Perceptron)\n',metric_multi_layer_perceptron)
    print (' AUC (Weighted Average)\n',metric_weighted_average)
    print (' AUC (Gradient Boosting - XGBoost)\n',metric_gradient_boosting)
    print (' AUC (Stacking)\n',metric_stacking)
    print (' AUC (Blending)\n',metric_blending)
    print('\nEnd Test Sample',n,'\n')


# # Testing the Base and Second Level Models on the Test Dataset

# In[45]:

def test_data():
    
    #Training the base models, and calculating AUC on the test data.
    #Selecting the data (Test Data)
    test_Y = test['y']
    test_X = test.drop(['y'],axis=1)
    
    #Computing the AUC and Predictions of all the base models on the test data parallely.
    auc_predict_test = (Parallel(n_jobs = -1)(delayed(function)(test_X,test_Y)                                          for function in cross_val_base_model_list))
    
    #Gradient Boosting (XGBoost)
    #The AUC error (Test Data)
    auc,predict_gradient_boosting = auc_predict_test[0][0],auc_predict_test[0][1]
    metric_gradient_boosting.append(auc)
    
    #Multi Layer Perceptron
    #The AUC (Test Data)
    predict_mlp=list()
    auc,predict_multi_layer_perceptron = auc_predict_test[1][0],auc_predict_test[1][1]
    metric_multi_layer_perceptron.append(auc)
    
    #predict_multi_layer_perceptron returns a list of lists containing the predictions, 
    #this cannot be converted to a dataframe.
    #This inner lists are converted to floats and then used to convert it to a dataframe.
    for i in predict_multi_layer_perceptron:
        predict_mlp.append(float(i))
    
    #Decision Tree)
    #The AUC (Test Data)
    auc,predict_decision_tree = auc_predict_test[2][0],auc_predict_test[2][1]
    metric_decision_tree.append(auc)
    
    #Random Forest (Deafult=10 Trees)
    #The AUC (Test Data)
    auc,predict_random_forest = auc_predict_test[3][0],auc_predict_test[3][1]
    metric_random_forest.append(auc)
    
    #Linear Regression
    #The AUC (Test Data)
    auc,predict_linear_regression = auc_predict_test[4][0],auc_predict_test[4][1]
    metric_linear_regression.append(auc)
    
    #Logistic Regression (Default=l2)
    #The AUC (Test Data)
    auc,predict_logistic_regression_L1 = auc_predict_test[5][0],auc_predict_test[5][1]
    metric_logistic_regression_L1.append(auc)
    
    #Logistic Regression-L2
    #The AUC (Test Data)
    auc,predict_logistic_regression_L2 = auc_predict_test[6][0],auc_predict_test[6][1]
    metric_logistic_regression_L2.append(auc)
    
    #Building a list that contains all the predictions of the base models.
    predict_list = [predict_gradient_boosting,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1,predict_mlp]
    global test_stack_X
    global test_raw_features_X
    global test_blend_X
    
    test_stack_X = build_data_frame(predict_list)
    #Converting the list of predictions into a dataframe.
    test_raw_features_X = test_raw_features_X.append(test_X,ignore_index = True)
    #Converting the above list of predictions and raw features (Concatenate) into a dataframe
    test_blend_X = pd.concat([test_raw_features_X, test_stack_X], axis = 1,ignore_index = True)

    #Performing a weighted average of all the base models and calculating the resulting AUC.
    auc,predict_weighted_average = weighted_average(test_stack_X,test_Y,weight)
    metric_weighted_average.append(auc)
        
    #Computing the AUC and Predictions of the Stacking and Blending models on the test data parallely.
    auc_predict_test_second_level = Parallel(n_jobs = -1)(delayed(function)(test_X, test_Y)                                                       for function,test_X in                                                       ((cross_val_second_level_model[0],test_stack_X),                                                       (cross_val_second_level_model[1],test_blend_X)))

    #Stacking (XGBoost - Gradient Boosting)
    auc,predict_stack = auc_predict_test_second_level[0][0],auc_predict_test_second_level[0][1]
    metric_stacking.append(auc)    

    #Blending (XGBoost - Gradient Boosting)
    auc,predict_blend = auc_predict_test_second_level[1][0],auc_predict_test_second_level[1][1]
    metric_blending.append(auc)
    
    


# In[46]:

sample_generation(1)


# In[47]:

#(Parallel(n_jobs=-1)(delayed(sample_generation)(n) for n in range(4)))


# In[ ]:



