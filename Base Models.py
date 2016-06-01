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


def gradient_boosting(train_X,train_Y,cross_val_X,cross_val_Y):
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
    dtrain = xgb.DMatrix(train_X,label=train_Y)
    dcross_val = xgb.DMatrix(cross_val_X,label=cross_val_Y)
    gradient_boosting = xgb.train(param, dtrain)
    #redict=XGB.predict(dcross_val)
    #predict[(predict>=0.5)]=1
    #predict[(predict<0.5)]=0
    predict=gradient_boosting.predict(dcross_val)
    #The AUC error (Cross Validation Data)
    auc=roc_auc_score(cross_val_Y,predict)
    return [auc, predict]

def multi_layer_perceptron(train_X,train_Y,cross_val_X,cross_val_Y):
    #Multi Layer Perceptron
    multi_layer_perceptron = Sequential()
    multi_layer_perceptron.add(Dense(output_dim=64, input_dim=20, init='uniform', activation='sigmoid'))
    multi_layer_perceptron.add(Dense(output_dim=1, input_dim=64,activation='sigmoid',))
    multi_layer_perceptron.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    multi_layer_perceptron.fit(train_X.as_matrix(), train_Y.as_matrix(), nb_epoch=5, batch_size=128)
    predict=multi_layer_perceptron.predict_on_batch(cross_val_X)
    #The AUC (Cross Validation Data)
    auc=roc_auc_score(cross_val_Y,predict)
    return [auc, predict]

def decision_tree(train_X,train_Y,cross_val_X,cross_val_Y):
    #Decision Tree
    decision_tree = DecisionTreeClassifier(max_depth=6)
    decision_tree.fit(train_X,train_Y)
    #The AUC (Cross Validation Data)
    predict=decision_tree.predict_proba(cross_val_X)[:,1]
    auc=roc_auc_score(cross_val_Y,predict)
    return [auc, predict]

def random_forest(train_X,train_Y,cross_val_X,cross_val_Y):
    #Random Forest (Deafult=10 Trees)
    random_forest = RandomForestClassifier()
    random_forest.fit(train_X,train_Y)
    #The AUC (Cross Validation Data)
    predict=random_forest.predict_proba(cross_val_X)[:,1]
    auc=roc_auc_score(cross_val_Y,predict)
    return [auc, predict]

def linear_regression(train_X,train_Y,cross_val_X,cross_val_Y):
    #Linear Regression
    predict=list()
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(train_X,train_Y)
    predict=linear_regression.predict(cross_val_X)
    #predict[(predict>=0.5)]=1
    #predict[(predict<0.5)]=0
    #The AUC (Cross Validation Data)
    auc=roc_auc_score(cross_val_Y,predict)
    return [auc, predict]

def logistic_regression(train_X,train_Y,cross_val_X,cross_val_Y,penalty):
    #Logistic Regression (Default=l2)
    logistic_regression = linear_model.LogisticRegression(penalty=penalty)
    logistic_regression.fit(train_X,train_Y)
    if(penalty=='l2'):
        predict=logistic_regression.predict_proba(cross_val_X)[:,1]
        #The AUC (Cross Validation Data)
        auc=roc_auc_score(cross_val_Y,predict)
    else:
        predict=logistic_regression.predict_proba(cross_val_X)[:,1]
        #The AUC (Cross Validation Data)
        auc=roc_auc_score(cross_val_Y,predict)
        
    return [auc, predict]

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
metric_a=list()
avg_a=0

#Cross Validation using Stratified K Fold
kf = StratifiedKFold(Data['y'], n_folds=5, shuffle=True)

for train_index, cross_val_index in kf:
    
    #Selecting the data
    train, cross_val = Data.iloc[train_index], Data.iloc[cross_val_index]
    train_Y=train['y']
    train_X=train.drop(['y'],axis=1)
    cross_val_Y=cross_val['y']
    cross_val_X=cross_val.drop(['y'],axis=1)
    
  
    #Gradient Boosting (XGBoost)
    #The AUC error (Cross Validation Data)
    [auc,predict_XGB]=gradient_boosting(train_X,train_Y,cross_val_X,cross_val_Y)
    metric_XGB.append(auc)

    
    #Multi Layer Perceptron
    #The AUC (Cross Validation Data)
    [auc,predict_multi_layer_perceptron]=(multi_layer_perceptron(train_X,train_Y,cross_val_X,cross_val_Y))
    metric_multi_layer_perceptron.append(auc)


    #Decision Tree)
    #The AUC (Cross Validation Data)
    [auc,predict_decision_tree]=decision_tree(train_X,train_Y,cross_val_X,cross_val_Y)
    metric_decision_tree.append(auc)
    
    
    #Random Forest (Deafult=10 Trees)
    #The AUC (Cross Validation Data)
    [auc,predict_random_forest]=random_forest(train_X,train_Y,cross_val_X,cross_val_Y)
    metric_random_forest.append(auc)
    
    #Scaling the data
    train_X=preprocessing.StandardScaler().fit_transform(train_X)
    cross_val_X=preprocessing.StandardScaler().fit_transform(cross_val_X)
    
    #Linear Regression
    #The AUC (Cross Validation Data)
    [auc,predict_linear_regression]=linear_regression(train_X,train_Y,cross_val_X,cross_val_Y)
    metric_linear_regression.append(auc)
    
    #Logistic Regression (Default=l2)
    #The AUC (Cross Validation Data)
    [auc,predict_logistic_regression_L2]=logistic_regression(train_X,train_Y,cross_val_X,cross_val_Y,'l2')
    metric_logistic_regression_L2.append(auc)
    
    #Logistic Regression-L1
    #The AUC (Cross Validation Data)
    [auc,predict_logistic_regression_L1]=logistic_regression(train_X,train_Y,cross_val_X,cross_val_Y,'l1')
    metric_logistic_regression_L1.append(auc)
    
    
avg_linear_regression=np.mean(metric_linear_regression)
avg_logistic_regression_L2=np.mean(metric_logistic_regression_L2)
avg_logistic_regression_L1=np.mean(metric_logistic_regression_L1)
avg_decision_tree=np.mean(metric_decision_tree)
avg_random_forest=np.mean(metric_random_forest)
avg_multi_layer_perceptron=np.mean(metric_multi_layer_perceptron)
avg_gradient_boosting=np.mean(metric_gradient_boosting)
avg_XGB=np.mean(metric_XGB)

print (' AUC (Linear Regression)\n',avg_linear_regression)
print (' AUC (Logistic Regression - L2)\n',avg_logistic_regression_L2)
print (' AUC (Logistic Regression - L1)\n',avg_logistic_regression_L1)
print (' AUC (Decision Tree)\n',avg_decision_tree)
print (' AUC (Random Forest)\n',avg_random_forest)
print (' AUC (Multi Layer Perceptron)\n',avg_multi_layer_perceptron)
print (' AUC (Gradient Boosting - XGBoost)\n',avg_XGB)