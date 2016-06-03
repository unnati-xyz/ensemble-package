
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold, KFold
from keras.layers import Dense, Activation, LSTM
from keras.models import Sequential
from keras.regularizers import l2, activity_l2
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from joblib import Parallel, delayed


# In[2]:

#Reading the data, into a Data Frame.
Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)

#Encoding the data, encoding the string values into numerical values.
encode = preprocessing.LabelEncoder()

#Selcting the columns of string data type
names = Data.select_dtypes(include=['object'])

#Function that encodes the string values to numerical values.
def enc(data,column):
    data[column] = encode.fit_transform(data[column])
    return data
for column in names:
        Data = enc(Data,column)
        
#Splitting the data into training and testing datasets
Data, test = train_test_split(Data, test_size = 0.05)

#Initializing two data frames that will be used as training data for the stacked model.
stack_X = pd.DataFrame() #The data frame will contain the predictions of the base models.
stack_Y = pd.DataFrame() #The data frame will contain the calss labels of the base models.

#Initializing two data frames that will be used as training data for the blending model.
blend_X = pd.DataFrame() #The data frames will contain the predictions and raw features  of the base models.
raw_features_X = pd.DataFrame() #The data frames will contain the raw features  of the data, which will be concatenated with the predictions.


# In[3]:

#This function is used to convert the predictions of the base models into a DataFrame.
def build_data_frame(data):
    data_frame = pd.DataFrame(data).T
    return data_frame


# In[4]:

#Defining the parameters for the XGBoost (Gradient Boosting) Algorithm.
def param_set():
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


# In[5]:

#This function is used to train the base and stacking models. Returns all the models to be used for further computations.
def train_models(train_X,train_Y,model):
    
    param = param_set()
    #Trains only the base models.
    if(model=='base'):
        
        #Gradient Boosting
        dtrain = xgb.DMatrix(train_X,label=train_Y)
        gradient_boosting = xgb.train(param, dtrain)
        
        #Multi Layer Perceptron
        multi_layer_perceptron = Sequential()
        multi_layer_perceptron.add(Dense(output_dim = 64, input_dim = 20, init = 'uniform', activation = 'sigmoid'))
        multi_layer_perceptron.add(Dense(output_dim = 1, input_dim = 64,activation = 'sigmoid',))
        multi_layer_perceptron.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])
        multi_layer_perceptron.fit(train_X.as_matrix(), train_Y.as_matrix(), nb_epoch = 5, batch_size = 128)
        
        #Decision Tree
        decision_tree = DecisionTreeClassifier(max_depth = 6)
        decision_tree.fit(train_X,train_Y)
        
        #Random Forest (Deafult=10 Trees)
        random_forest = RandomForestClassifier()
        random_forest.fit(train_X,train_Y)
        
        #Scaling the data
        train_X = preprocessing.StandardScaler().fit_transform(train_X) 
        
        #Linear Regression
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(train_X,train_Y)
        
        #Logistic Regression (L1)
        logistic_regression_L1 = linear_model.LogisticRegression(penalty = 'l1')
        logistic_regression_L1.fit(train_X,train_Y)
        
        #Logistic Regression (L2)
        logistic_regression_L2 = linear_model.LogisticRegression(penalty = 'l2')
        logistic_regression_L2.fit(train_X,train_Y)
        
        #Returns a dictionary containing the model names and their respective models.
        return {'XGBoost':gradient_boosting,'Multi Layer Perceptron':multi_layer_perceptron,'Decision Tree':decision_tree,
           'Random Forest':random_forest,'Linear Regression':linear_regression,'L1':logistic_regression_L1,
            'L2':logistic_regression_L2}
    
    #Trains the stacking model (Gradient Boosting - XGBoost)
    elif(model == 'stack'):
        
        dtrain = xgb.DMatrix(train_X,label = train_Y)
        stack = xgb.train(param, dtrain)
        return {'Stack':stack}
    
    #Trains the blending model (Gradient Boosting - XGBoost)
    else:
        
        dtrain = xgb.DMatrix(train_X,label = train_Y)
        blend = xgb.train(param, dtrain)
        return {'Blend':blend}


# In[6]:

#Function calculates area under the curve and predictions on the given data, for the model specified.
def cross_validation(model_name,model,cross_val_X,cross_val_Y):
    
    if(model_name == 'Gradient Boosting' or model_name == 'Linear Regression'):
        
        predict = model.predict(cross_val_X)
        
    elif(model_name == 'Multi Layer Perceptron'):
        
        predict = model.predict_on_batch(cross_val_X)
    else:
        
        predict = model.predict_proba(cross_val_X)[:,1]
        
    auc = roc_auc_score(cross_val_Y,predict)
    
    return[auc,predict]


# In[7]:

#Initialzing the variables that will be used to calculate the area under the curve. (cross Validation Data)
metric_linear_regression=list()
avg_linear_regeression=0
metric_logistic_regression_L2=list()
avg_logistic_regression_L2=0
metric_logistic_regression_L1=list()
avg_logistic_regression_L1=0
metric_decision_tree=list()
avg_decision_tree=0
metric_random_forest=list()
avg_random_forest=0
metric_multi_layer_perceptron=list()
avg_multi_layer_perceptron=0
metric_gradient_boosting=list()
avg_gradient_boosting=0


# In[8]:

#Cross Validation using Stratified K Fold
kf = StratifiedKFold(Data['y'], n_folds=5, shuffle=True)


# In[9]:

#Training the base models, and calculating AUC on the cross validation data.
for train_index, cross_val_index in kf:
    
    #Selecting the data (Traing Data & Cross Validation Data)
    train, cross_val = Data.iloc[train_index], Data.iloc[cross_val_index]
    train_Y=train['y']
    train_X=train.drop(['y'],axis=1)
    cross_val_Y=cross_val['y']
    cross_val_X=cross_val.drop(['y'],axis=1)
    scale=preprocessing.StandardScaler()
    
    #Training the base models, the resulting model names and models are stored in the variable model in the from of a dictionary.
    model=train_models(train_X,train_Y,'base')
  
    #Gradient Boosting (XGBoost)
    #The AUC error (Cross Validation Data)
    [auc,predict_gradient_boosting]=cross_validation('Gradient Boosting',model['XGBoost'],xgb.DMatrix(cross_val_X,label=cross_val_Y),cross_val_Y)
    metric_gradient_boosting.append(auc)

    #Multi Layer Perceptron
    #The AUC (Cross Validation Data)
    predict_mlp=list()
    [auc,predict_multi_layer_perceptron]=cross_validation('Multi Layer Perceptron',model['Multi Layer Perceptron'],cross_val_X,cross_val_Y)
    metric_multi_layer_perceptron.append(auc)
    #predict_multi_layer_perceptron returns a list of lists containing the predictions, this cannot be converted to a dataframe.
    #This inner lists are converted to floats and then used to convert it to a dataframe.
    for i in predict_multi_layer_perceptron:
        predict_mlp.append(float(i))
    
    #Decision Tree)
    #The AUC (Cross Validation Data)
    [auc,predict_decision_tree]=cross_validation('Decision Tree',model['Decision Tree'],cross_val_X,cross_val_Y)
    metric_decision_tree.append(auc)
    
    #Random Forest (Deafult=10 Trees)
    #The AUC (Cross Validation Data)
    [auc,predict_random_forest]=cross_validation('Random Forest',model['Random Forest'],cross_val_X,cross_val_Y)
    metric_random_forest.append(auc)
    
    #Scaling the cross validation data.
    cross_val_X=scale.fit_transform(cross_val_X)
    
    #Linear Regression
    #The AUC (Cross Validation Data)
    [auc,predict_linear_regression]=cross_validation('Linear Regression',model['Linear Regression'],cross_val_X,cross_val_Y)
    metric_linear_regression.append(auc)
    
    #Logistic Regression (Default=l2)
    #The AUC (Cross Validation Data)
    [auc,predict_logistic_regression_L2]=cross_validation('L2',model['L2'],cross_val_X,cross_val_Y)
    metric_logistic_regression_L2.append(auc)
    
    #Logistic Regression-L1
    #The AUC (Cross Validation Data)
    [auc,predict_logistic_regression_L1]=cross_validation('L1',model['L1'],cross_val_X,cross_val_Y)
    metric_logistic_regression_L1.append(auc)
    
    #Building a list that contains all the predictions of the base models.
    predict_list=[predict_gradient_boosting,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1,predict_mlp]
    
    #Rescaling the cross validation data back to its original values.
    cross_val_X=scale.inverse_transform(cross_val_X)
    
    #Converting the above list of predictions into a dataframe, which will be used to train the stacking model.
    stack_Y=stack_Y.append(cross_val_Y.tolist())
    stack_X=stack_X.append(build_data_frame(predict_list))
    
    #Building a list that contains all the raw features used as cross validation data for the base models.
    raw_features_X=raw_features_X.append(cross_val_X.tolist())


# In[10]:

#Calculating the average AUC across all the AUC computed on the cross validation folds.
avg_linear_regression=np.mean(metric_linear_regression)
avg_logistic_regression_L2=np.mean(metric_logistic_regression_L2)
avg_logistic_regression_L1=np.mean(metric_logistic_regression_L1)
avg_decision_tree=np.mean(metric_decision_tree)
avg_random_forest=np.mean(metric_random_forest)
avg_multi_layer_perceptron=np.mean(metric_multi_layer_perceptron)
avg_gradient_boosting=np.mean(metric_gradient_boosting)


# In[11]:

#Printing the AUC for the base models.
print (' AUC (Linear Regression)\n',avg_linear_regression)
print (' AUC (Logistic Regression - L2)\n',avg_logistic_regression_L2)
print (' AUC (Logistic Regression - L1)\n',avg_logistic_regression_L1)
print (' AUC (Decision Tree)\n',avg_decision_tree)
print (' AUC (Random Forest)\n',avg_random_forest)
print (' AUC (Multi Layer Perceptron)\n',avg_multi_layer_perceptron)
print (' AUC (Gradient Boosting - XGBoost)\n',avg_gradient_boosting)


# In[12]:

#Training the stacking model(XGBoost-Gradient Boosting)
model_stack=train_models(stack_X,stack_Y,'stack')

#Converting the above list of predictions and raw features (Concatenate) into a dataframe, which will be used to train the blending model.
blend_X=pd.concat([raw_features_X, stack_X], axis=1,ignore_index=True)

#Training the blending model(XGBoost-Gradient Boosting)
model_blend=train_models(blend_X,stack_Y,'blend')


# In[13]:

#Initialzing the variables that will be used to calculate the area under the curve. (Test Data)
metric_logistic_regression_L2=list()
metric_logistic_regression_L1=list()
metric_decision_tree=list()
metric_random_forest=list()
metric_multi_layer_perceptron=list()
metric_gradient_boosting=list()
metric_stack=list()
metric_blend=list()
blend_X = pd.DataFrame()
raw_features_X = pd.DataFrame()


# In[14]:

#Calculating AUC for all the models (Base Models & Stack Model) on the test data.

#Selecting the test data
test_Y=test['y']
test_X=test.drop(['y'],axis=1)
scale=preprocessing.StandardScaler()
    
#Gradient Boosting (XGBoost)
#The AUC error (Test Data)
[auc,predict_XGB]=cross_validation('Gradient Boosting',model['XGBoost'],xgb.DMatrix(test_X,label=test_Y),test_Y)
metric_gradient_boosting=(auc)

    
#Multi Layer Perceptron
#The AUC (Test Data)
predict_mlp=list()
[auc,predict_multi_layer_perceptron]=cross_validation('Multi Layer Perceptron',model['Multi Layer Perceptron'],test_X,test_Y)
metric_multi_layer_perceptron=(auc)

#predict_multi_layer_perceptron returns a list of lists containing the predictions, this cannot be converted to a dataframe.
#This inner lists are converted to floats and then used to convert it to a dataframe.
for i in predict_multi_layer_perceptron:
        predict_mlp.append(float(i))


#Decision Tree)
#The AUC (Test Data)
[auc,predict_decision_tree]=cross_validation('Decision Tree',model['Decision Tree'],test_X,test_Y)
metric_decision_tree=(auc)
    
    
#Random Forest (Deafult=10 Trees)
#The AUC (Test Data)
[auc,predict_random_forest]=cross_validation('Random Forest',model['Random Forest'],test_X,test_Y)
metric_random_forest=(auc)
    
test_X=scale.fit_transform(test_X)
#Linear Regression
#The AUC (Test Data)
[auc,predict_linear_regression]=cross_validation('Linear Regression',model['Linear Regression'],test_X,test_Y)
metric_linear_regression=(auc)
    
#Logistic Regression (Default=l2)
#The AUC (Test Data)
[auc,predict_logistic_regression_L2]=cross_validation('L2',model['L2'],test_X,test_Y)
metric_logistic_regression_L2=(auc)

#Logistic Regression-L1
#The AUC (Test Data)
[auc,predict_logistic_regression_L1]=cross_validation('L1',model['L1'],test_X,test_Y)
metric_logistic_regression_L1=(auc)

#Building a list that contains all the predictions of the base models.
predict_list=[predict_XGB,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1,predict_mlp]

#Rescaling the test data back to its original values.
test_X=scale.inverse_transform(test_X)
    
#Stacking (XGBoost - Gradient Boosting)
dstack_X=build_data_frame(predict_list) #Converting the list of predictions into a dataframe.
[auc,predict_stack]=cross_validation('Gradient Boosting',model_stack['Stack'],xgb.DMatrix(dstack_X,label=test_Y),test_Y)
metric_stack=(auc)    

#Blending (XGBoost - Gradient Boosting)
raw_features_X=raw_features_X.append(test_X.tolist())
blend_X=pd.concat([raw_features_X, dstack_X], axis=1,ignore_index=True)#Converting the above list of predictions and raw features (Concatenate) into a dataframe
[auc,predict_blend]=cross_validation('Gradient Boosting',model_blend['Blend'],xgb.DMatrix(blend_X,label=test_Y),test_Y)
metric_blend=(auc) 


# In[15]:

print (' AUC (Linear Regression)\n',metric_linear_regression)
print (' AUC (Logistic Regression - L2)\n',metric_logistic_regression_L2)
print (' AUC (Logistic Regression - L1)\n',metric_logistic_regression_L1)
print (' AUC (Decision Tree)\n',metric_decision_tree)
print (' AUC (Random Forest)\n',metric_random_forest)
print (' AUC (Multi Layer Perceptron)\n',metric_multi_layer_perceptron)
print (' AUC (Gradient Boosting - XGBoost)\n',metric_gradient_boosting)
print (' AUC (Stacking)\n',metric_stack)
print (' AUC (Blending)\n',metric_blend)

