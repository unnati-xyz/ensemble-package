
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
        
#Splitting the data into training and testing datasets (Stratified Split)
Data, test = train_test_split(Data, test_size = 0.1,stratify=Data['y'])

#The list of base models.
base_model_list=['XGBoost','Multi Layer Perceptron','Decision Tree','Random Forest','Linear Regression','L1','L2']

#The list of second level models.
second_level_model_list=['Stack','Blend']

model_name_list=list() #A list that contains both the models and the model names. Will be used during cross validation.
second_level_model_name_list=list()

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
def train_base_models(train_X,train_Y,model):
    
    param = param_set()
    
    if(model=='XGBoost'):
        
        #Gradient Boosting
        dtrain = xgb.DMatrix(train_X,label=train_Y)
        gradient_boosting = xgb.train(param, dtrain)
        return gradient_boosting
       
    elif(model=='Multi Layer Perceptron'):
        
        #Multi Layer Perceptron
        multi_layer_perceptron = Sequential()
        multi_layer_perceptron.add(Dense(output_dim = 64, input_dim = 20, init = 'uniform', activation = 'sigmoid'))
        multi_layer_perceptron.add(Dense(output_dim = 1, input_dim = 64,activation = 'sigmoid',))
        multi_layer_perceptron.compile(optimizer = 'rmsprop',loss = 'binary_crossentropy',metrics = ['accuracy'])
        multi_layer_perceptron.fit(train_X.as_matrix(), train_Y.as_matrix(), nb_epoch = 5, batch_size = 128)
        return multi_layer_perceptron
        
    elif(model=='Decision Tree'):
        #Decision Tree
        decision_tree = DecisionTreeClassifier(max_depth = 6)
        decision_tree.fit(train_X,train_Y)
        return decision_tree
        
    elif(model=='Random Forest'):
        
        #Random Forest (Deafult=10 Trees)
        random_forest = RandomForestClassifier()
        random_forest.fit(train_X,train_Y)
        return random_forest
        
    elif(model=='Linear Regression'):
        
        #Scaling the data
        train_X = preprocessing.StandardScaler().fit_transform(train_X)
        
        #Linear Regression
        linear_regression = linear_model.LinearRegression()
        linear_regression.fit(train_X,train_Y)
        return linear_regression
        
    elif(model=='L1'):
        
        #Scaling the data
        train_X = preprocessing.StandardScaler().fit_transform(train_X)
        
        #Logistic Regression (L1)
        logistic_regression_L1 = linear_model.LogisticRegression(penalty = 'l1')
        logistic_regression_L1.fit(train_X,train_Y)
        return logistic_regression_L1
        
    elif(model=='L2'):
       
        #Scaling the data
        train_X = preprocessing.StandardScaler().fit_transform(train_X)
        
        #Logistic Regression (L2)
        logistic_regression_L2 = linear_model.LogisticRegression(penalty = 'l2')
        logistic_regression_L2.fit(train_X,train_Y)
        return logistic_regression_L2    


# In[6]:

def train_second_level_models(train_X,train_Y,model):
    
    param = param_set()
    
    #Trains the stacking model (Gradient Boosting - XGBoost)
    if(model == 'Stack'):
        
        dtrain = xgb.DMatrix(train_X,label = train_Y)
        stack = xgb.train(param, dtrain)
        return stack
    
    #Trains the blending model (Gradient Boosting - XGBoost)
    elif(model == 'Blend'):
        
        dtrain = xgb.DMatrix(train_X,label = train_Y)
        blend = xgb.train(param, dtrain)
        return blend


# In[7]:

#Function calculates area under the curve and predictions on the given data, for the specified model.
def cross_validation(model_name,model,cross_val_X,cross_val_Y):
    
    if(model_name == 'XGBoost'): 
        
        predict = model.predict(xgb.DMatrix(cross_val_X,label=cross_val_Y))
        
    elif(model_name == 'Linear Regression'):
        
        cross_val_X=preprocessing.StandardScaler().fit_transform(cross_val_X)
        predict = model.predict(cross_val_X)
        
    elif(model_name == 'L1' or model_name == 'L2'):
        
        cross_val_X=preprocessing.StandardScaler().fit_transform(cross_val_X)
        predict = model.predict_proba(cross_val_X)[:,1]
        
    elif(model_name == 'Multi Layer Perceptron'):
        
        predict = model.predict_on_batch(cross_val_X)
    else:
        
        predict = model.predict_proba(cross_val_X)[:,1]
        
    auc = roc_auc_score(cross_val_Y,predict)
    
    return [auc,predict]


# In[8]:

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


# In[9]:

#Cross Validation using Stratified K Fold
kf = StratifiedKFold(Data['y'], n_folds=2, shuffle=True)


# In[10]:

#Training the base models, and calculating AUC on the cross validation data.
for train_index, cross_val_index in kf:
    
    #Selecting the data (Traing Data & Cross Validation Data)
    train, cross_val = Data.iloc[train_index], Data.iloc[cross_val_index]
    train_Y=train['y']
    train_X=train.drop(['y'],axis=1)
    cross_val_Y=cross_val['y']
    cross_val_X=cross_val.drop(['y'],axis=1)
    
    #Training the base models parallely, the resulting models are stored in the variable models in the form of a list.
    models=(Parallel(n_jobs=-1)(delayed(train_base_models)(train_X, train_Y, model) for model in base_model_list))
    
    #The list will contain both the models and the model names. The list is used in the next step.
    for i in range(len(base_model_list)):
        model_name_list.append((base_model_list[i],models[i]))
        
    #Computing the AUC and Predictions of all the base models on the cross validation data parallely.
    auc_predict_cross_val=(Parallel(n_jobs=-1)(delayed(cross_validation)(model_name,model,cross_val_X,cross_val_Y) for model_name,model in model_name_list))
    
    #Gradient Boosting (XGBoost)
    #The AUC error (Cross Validation Data)
    auc,predict_gradient_boosting=auc_predict_cross_val[0][0],auc_predict_cross_val[0][1]
    metric_gradient_boosting.append(auc)
    
    #Multi Layer Perceptron
    #The AUC (Cross Validation Data)
    predict_mlp=list()
    auc,predict_multi_layer_perceptron=auc_predict_cross_val[1][0],auc_predict_cross_val[1][1]
    metric_multi_layer_perceptron.append(auc)
    
    #predict_multi_layer_perceptron returns a list of lists containing the predictions, this cannot be converted to a dataframe.
    #This inner lists are converted to floats and then used to convert it to a dataframe.
    for i in predict_multi_layer_perceptron:
        predict_mlp.append(float(i))
    
    #Decision Tree)
    #The AUC (Cross Validation Data)
    auc,predict_decision_tree=auc_predict_cross_val[2][0],auc_predict_cross_val[2][1]
    metric_decision_tree.append(auc)
    
    #Random Forest (Deafult=10 Trees)
    #The AUC (Cross Validation Data)
    auc,predict_random_forest=auc_predict_cross_val[3][0],auc_predict_cross_val[3][1]
    metric_random_forest.append(auc)
    
    #Linear Regression
    #The AUC (Cross Validation Data)
    auc,predict_linear_regression=auc_predict_cross_val[4][0],auc_predict_cross_val[4][1]
    metric_linear_regression.append(auc)
    
    #Logistic Regression (Default=l2)
    #The AUC (Cross Validation Data)
    auc,predict_logistic_regression_L1=auc_predict_cross_val[5][0],auc_predict_cross_val[5][1]
    metric_logistic_regression_L1.append(auc)
    
    
    #Logistic Regression-L2
    #The AUC (Cross Validation Data)
    auc,predict_logistic_regression_L2=auc_predict_cross_val[6][0],auc_predict_cross_val[6][1]
    metric_logistic_regression_L2.append(auc)
    
    #Building a list that contains all the predictions of the base models.
    predict_list=[predict_gradient_boosting,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1,predict_mlp]
    
    #Converting the above list of predictions into a dataframe, which will be used to train the stacking model.
    stack_X=stack_X.append(build_data_frame(predict_list))
    
    #Building a list that contains all the raw features, used as cross validation data for the base models.
    raw_features_X=raw_features_X.append(cross_val_X,ignore_index=True)
                              
    break


# In[11]:

#Calculating the average AUC across all the AUC computed on the cross validation folds.
avg_linear_regression=np.mean(metric_linear_regression)
avg_logistic_regression_L2=np.mean(metric_logistic_regression_L2)
avg_logistic_regression_L1=np.mean(metric_logistic_regression_L1)
avg_decision_tree=np.mean(metric_decision_tree)
avg_random_forest=np.mean(metric_random_forest)
avg_multi_layer_perceptron=np.mean(metric_multi_layer_perceptron)
avg_gradient_boosting=np.mean(metric_gradient_boosting)


# In[12]:

#Printing the AUC for the base models.
print (' AUC (Linear Regression)\n',avg_linear_regression)
print (' AUC (Logistic Regression - L2)\n',avg_logistic_regression_L2)
print (' AUC (Logistic Regression - L1)\n',avg_logistic_regression_L1)
print (' AUC (Decision Tree)\n',avg_decision_tree)
print (' AUC (Random Forest)\n',avg_random_forest)
print (' AUC (Multi Layer Perceptron)\n',avg_multi_layer_perceptron)
print (' AUC (Gradient Boosting - XGBoost)\n',avg_gradient_boosting)


# In[13]:

#Converting the above list of predictions and raw features (Concatenate) into a dataframe, which will be used to train the blending model.
blend_X=pd.concat([raw_features_X, stack_X], axis=1,ignore_index=True)

#Training the Stacking and Blending models parallely using the predictions of base models on the cross validation data.
model_second_level = Parallel(n_jobs=-1)(delayed(train_second_level_models)(train_X, cross_val_Y, model) for train_X,model in ((stack_X,second_level_model_list[0]),(blend_X,second_level_model_list[1])))


# In[14]:

#Initialzing the variables that will be used to calculate the area under the curve. (Test Data)
metric_linear_regression=list()
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


# In[15]:

#Calculating AUC for all the models (Base Models & Stack Model) on the test data.

#Selecting the test data
test_Y=test['y']
test_X=test.drop(['y'],axis=1)

#Computing the AUC and Predictions of all the base models on the test data parallely.
auc_predict_test_base=(Parallel(n_jobs=-1)(delayed(cross_validation)(model_name,model,test_X,test_Y) for model_name,model in model_name_list))
    
#Gradient Boosting (XGBoost)
#The AUC error (Cross Validation Data)
auc,predict_gradient_boosting=auc_predict_test_base[0][0],auc_predict_test_base[0][1]
metric_gradient_boosting.append(auc)
    
#Multi Layer Perceptron
#The AUC (Cross Validation Data)
predict_mlp=list()
auc,predict_multi_layer_perceptron=auc_predict_test_base[1][0],auc_predict_test_base[1][1]
metric_multi_layer_perceptron.append(auc)
#predict_multi_layer_perceptron returns a list of lists containing the predictions, this cannot be converted to a dataframe.
#This inner lists are converted to floats and then used to convert it to a dataframe.
for i in predict_multi_layer_perceptron:
    predict_mlp.append(float(i))
    
#Decision Tree
#The AUC (Cross Validation Data)
auc,predict_decision_tree=auc_predict_test_base[2][0],auc_predict_test_base[2][1]
metric_decision_tree.append(auc)
    
#Random Forest (Deafult=10 Trees)
#The AUC (Cross Validation Data)
auc,predict_random_forest=auc_predict_test_base[3][0],auc_predict_test_base[3][1]
metric_random_forest.append(auc)
    
#Linear Regression
#The AUC (Cross Validation Data)
auc,predict_linear_regression=auc_predict_test_base[4][0],auc_predict_test_base[4][1]
metric_linear_regression.append(auc)
    
#Logistic Regression (Default=l2)
#The AUC (Cross Validation Data)
auc,predict_logistic_regression_L1=auc_predict_test_base[5][0],auc_predict_test_base[5][1]
metric_logistic_regression_L1.append(auc)
    
    
#Logistic Regression-L2
#The AUC (Cross Validation Data)
auc,predict_logistic_regression_L2=auc_predict_test_base[6][0],auc_predict_test_base[6][1]
metric_logistic_regression_L2.append(auc)
    
#Building a list that contains all the predictions of the base models.
predict_list=[predict_gradient_boosting,predict_decision_tree,predict_random_forest, 
                               predict_linear_regression,predict_logistic_regression_L2,
                               predict_logistic_regression_L1,predict_mlp]
    


dstack_X=build_data_frame(predict_list)#Converting the list of predictions into a dataframe.
raw_features_X=raw_features_X.append(test_X,ignore_index=True)
blend_X=pd.concat([raw_features_X, dstack_X], axis=1,ignore_index=True)#Converting the above list of predictions and raw features (Concatenate) into a dataframe

#Computing the AUC and Predictions of the Stacking and Blending models on the test data parallely.
auc_predict_test_second_level=Parallel(n_jobs=-1)(delayed(cross_validation)('XGBoost',model,test_X, test_Y) for test_X,model in ((dstack_X,model_second_level[0]),(blend_X,model_second_level[1])))

#Stacking (XGBoost - Gradient Boosting)
auc,predict_stack=auc_predict_test_second_level[0][0],auc_predict_test_second_level[0][1]
metric_stack=(auc)    

#Blending (XGBoost - Gradient Boosting)
auc,predict_blend=auc_predict_test_second_level[1][0],auc_predict_test_second_level[1][1]
metric_blend=(auc) 


# In[16]:

print (' AUC (Linear Regression)\n',metric_linear_regression)
print (' AUC (Logistic Regression - L2)\n',metric_logistic_regression_L2)
print (' AUC (Logistic Regression - L1)\n',metric_logistic_regression_L1)
print (' AUC (Decision Tree)\n',metric_decision_tree)
print (' AUC (Random Forest)\n',metric_random_forest)
print (' AUC (Multi Layer Perceptron)\n',metric_multi_layer_perceptron)
print (' AUC (Gradient Boosting - XGBoost)\n',metric_gradient_boosting)
print (' AUC (Stacking)\n',metric_stack)
print (' AUC (Blending)\n',metric_blend)


# In[ ]:



