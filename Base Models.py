
# coding: utf-8

# In[791]:

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


# In[792]:

#Reading the data
Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)

#Encoding the data, encoding the string values into numerical values
encode = preprocessing.LabelEncoder()

#Selcting the columns of string data type
names=Data.select_dtypes(include=['object'])

#Function that encodes the string values to numerical values
def enc(data,column):
    data[column] = encode.fit_transform(data[column])
    return data
for column in names:
        Data=enc(Data,column)

Data, test = train_test_split(Data, test_size = 0.1)
#Data.job = encode.fit_transform(Data.job)
#Data.marital = encode.fit_transform(Data.marital)
#Data.education = encode.fit_transform(Data.education)
#Data.default = encode.fit_transform(Data.default)
#Data.housing = encode.fit_transform(Data.housing)
#Data.loan = encode.fit_transform(Data.loan)
#Data.contact = encode.fit_transform(Data.contact)
#Data.month = encode.fit_transform(Data.month)
#Data.day_of_week = encode.fit_transform(Data.day_of_week)
#Data.poutcome = encode.fit_transform(Data.poutcome)
#Data.y = encode.fit_transform(Data.y)


# In[793]:

def build_data_frame(data):
    data_frame=pd.DataFrame(data).T
    return data_frame


# In[794]:

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
    param['min_child_weight']=1
    param['max_delta_step'] = 0
    param['subsample']= 1
    param['colsample_bytree']=1
    param['silent'] = 1
    param['seed'] = 0
    param['base_score'] = 0.5
    param['lambda_bias']=1
    return param


# In[795]:

def train_models(train_X,train_Y,model):
    param = param_set()
    if(model=='base'):
        #Gradient Boosting
        dtrain = xgb.DMatrix(train_X,label=train_Y)
        gradient_boosting = xgb.train(param, dtrain)
        
        #Multi Layer Perceptron
        multi_layer_perceptron = Sequential()
        multi_layer_perceptron.add(Dense(output_dim=64, input_dim=20, init='uniform', activation='sigmoid'))
        multi_layer_perceptron.add(Dense(output_dim=1, input_dim=64,activation='sigmoid',))
        multi_layer_perceptron.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        multi_layer_perceptron.fit(train_X.as_matrix(), train_Y.as_matrix(), nb_epoch=5, batch_size=128)
        
        #Decision Tree
        decision_tree = DecisionTreeClassifier(max_depth=6)
        decision_tree.fit(train_X,train_Y)
        
        #Random Forest (Deafult=10 Trees)
        random_forest = RandomForestClassifier()
        random_forest.fit(train_X,train_Y)
        
        #Scaling the data
        train_X=preprocessing.StandardScaler().fit_transform(train_X) 
        
        #Linear Regression
        predict=list()
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(train_X,train_Y)
        #Logistic Regression (L1)
        
        logistic_regression_L1 = linear_model.LogisticRegression(penalty='l1')
        logistic_regression_L1.fit(train_X,train_Y)
        #Logistic Regression (L2)
        
        logistic_regression_L2 = linear_model.LogisticRegression(penalty='l2')
        logistic_regression_L2.fit(train_X,train_Y)
        
        return {'XGBoost':gradient_boosting,'Multi Layer Perceptron':multi_layer_perceptron,'Decision Tree':decision_tree,
           'Random Forest':random_forest,'Linear Regression':linear_regression,'L1':logistic_regression_L1,
            'L2':logistic_regression_L2}
    else:
        dtrain = xgb.DMatrix(train_X,label=train_Y)
        stack = xgb.train(param, dtrain)
        return {'Stack':stack}
     
    


# In[796]:

def cross_validation(model_name,model,cross_val_X,cross_val_Y):
    if(model_name=='Gradient Boosting' or model_name=='Linear Regression'):
        predict=model.predict(cross_val_X)
        
    elif(model_name=='Multi Layer Perceptron'):
        predict=model.predict_on_batch(cross_val_X)
    else:
        predict=model.predict_proba(cross_val_X)[:,1]
    auc=roc_auc_score(cross_val_Y,predict)
    return[auc,predict]
    


# In[797]:

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
metric_XGB=list()
avg_XGB=0
metric_stack=list()
avg_stack=0


# In[798]:

#Cross Validation using Stratified K Fold
kf = StratifiedKFold(Data['y'], n_folds=5, shuffle=True)


# In[ ]:

for train_index, cross_val_index in kf:
    
    #Selecting the data
    train, cross_val = Data.iloc[train_index], Data.iloc[cross_val_index]
    train_Y=train['y']
    train_X=train.drop(['y'],axis=1)
    cross_val_Y=cross_val['y']
    cross_val_X=cross_val.drop(['y'],axis=1)
    
    model=train_models(train_X,train_Y,'base')
  
    #Gradient Boosting (XGBoost)
    #The AUC error (Cross Validation Data)
    [auc,predict_XGB]=cross_validation('Gradient Boosting',model['XGBoost'],xgb.DMatrix(cross_val_X,label=cross_val_Y),cross_val_Y)
    metric_XGB.append(auc)

    
    #Multi Layer Perceptron
    #The AUC (Cross Validation Data)
    [auc,predict_multi_layer_perceptron]=cross_validation('Multi Layer Perceptron',model['Multi Layer Perceptron'],cross_val_X,cross_val_Y)
    metric_multi_layer_perceptron.append(auc)


    #Decision Tree)
    #The AUC (Cross Validation Data)
    [auc,predict_decision_tree]=cross_validation('Decision Tree',model['Decision Tree'],cross_val_X,cross_val_Y)
    metric_decision_tree.append(auc)
    
    
    #Random Forest (Deafult=10 Trees)
    #The AUC (Cross Validation Data)
    [auc,predict_random_forest]=random_forest(train_X,train_Y,cross_val_X,cross_val_Y)
    metric_random_forest.append(auc)
    
    cross_val_X=preprocessing.StandardScaler().fit_transform(cross_val_X)
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
    
    predict_list=[predict_XGB,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1]
    
    #Stacking (XGBoost - Gradient Boosting)
    stack_X=build_data_frame(predict_list)
    model_stack=train_models(stack_X,cross_val_Y,'stack')


# In[ ]:

avg_linear_regression=np.mean(metric_linear_regression)
avg_logistic_regression_L2=np.mean(metric_logistic_regression_L2)
avg_logistic_regression_L1=np.mean(metric_logistic_regression_L1)
avg_decision_tree=np.mean(metric_decision_tree)
avg_random_forest=np.mean(metric_random_forest)
avg_multi_layer_perceptron=np.mean(metric_multi_layer_perceptron)
avg_XGB=np.mean(metric_XGB)


# In[ ]:

print (' AUC (Linear Regression)\n',avg_linear_regression)
print (' AUC (Logistic Regression - L2)\n',avg_logistic_regression_L2)
print (' AUC (Logistic Regression - L1)\n',avg_logistic_regression_L1)
print (' AUC (Decision Tree)\n',avg_decision_tree)
print (' AUC (Random Forest)\n',avg_random_forest)
print (' AUC (Multi Layer Perceptron)\n',avg_multi_layer_perceptron)
print (' AUC (Gradient Boosting - XGBoost)\n',avg_XGB)


# In[ ]:



