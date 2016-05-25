import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, preprocessing
from sklearn.preprocessing import Imputer, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedKFold
from keras.layers import Dense, Activation
from keras.models import Sequential

encode = preprocessing.LabelEncoder()
Data = pd.read_csv('/home/prajwal/Desktop/bank-additional/bank-additional-full.csv',delimiter=';',header=0)
#print (Data)
Data.job = encode.fit_transform(Data.job)
Data.marital = encode.fit_transform(Data.marital)
Data.education = encode.fit_transform(Data.education)
Data.default = encode.fit_transform(Data.default)
Data.housing = encode.fit_transform(Data.housing)
Data.loan = encode.fit_transform(Data.loan)
Data.contact = encode.fit_transform(Data.contact)
Data.month = encode.fit_transform(Data.month)
Data.day_of_week = encode.fit_transform(Data.day_of_week)
Data.poutcome = encode.fit_transform(Data.poutcome)
Data.y = encode.fit_transform(Data.y)

mean_LinearRegression=list()
avg_LinearRegeression=0
mean_L2=list()
avg_L2=0
mean_L1=list()
avg_L1=0
mean_DT=list()
avg_DT=0
mean_RF=list()
avg_RF=0
mean_MLP=list()
avg_MLP=0

variance_L2=list()
variance_L1=list()
variance_DT=list()
variance_RF=list()
variance_MLP=list()

kf = StratifiedKFold(Data['y'], n_folds=5,shuffle=True)

for train_index, cross_val_index in kf:
    train, cross_val = Data.iloc[train_index], Data.iloc[cross_val_index]
    train_Y = train['y']
    train_X = train.drop(['y'], axis=1)
    cross_val_Y = cross_val['y']
    cross_val_X = cross_val.drop(['y'], axis=1)
    predict = list()
       
    #Multi Layer Perceptron
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=20, init='uniform', activation='sigmoid'))
    model.add(Dense(output_dim=1, input_dim=64,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['accuracy'])
    model.fit(train_X.as_matrix(), train_Y.as_matrix(), nb_epoch=5, batch_size=32)
    eval=model.test_on_batch(cross_val_X.as_matrix(), cross_val_Y.as_matrix(), sample_weight=None)
    mean_MLP.append(eval[0])
    variance_MLP.append(eval[1])

    # Decision Tree
    model = DecisionTreeClassifier()
    model.fit(train_X, train_Y)
    # The mean square error
    mean_DT.append(np.mean((model.predict(cross_val_X) - cross_val_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    variance_DT.append(model.score(cross_val_X, cross_val_Y))

    # Random Forest (Deafult=10 Trees)
    model = RandomForestClassifier()
    model.fit(train_X, train_Y)
    # The mean square error
    mean_RF.append(np.mean((model.predict(cross_val_X) - cross_val_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    variance_RF.append(model.score(cross_val_X, cross_val_Y))

    #Scaling the data
    train_X = preprocessing.StandardScaler().fit_transform(train_X)
    cross_val_X = preprocessing.StandardScaler().fit_transform(cross_val_X)

    # Linear Regression
    regr = linear_model.LinearRegression()
    regr.fit(train_X, train_Y)
    predict = regr.predict(cross_val_X)
    predict[(predict >= 0.5)] = 1
    predict[(predict < 0.5)] = 0
    mean_LinearRegression.append((np.mean((predict - cross_val_Y) ** 2)))

    # Logistic Regression (Default=L2)
    regr = linear_model.LogisticRegression(penalty='l2')
    regr.fit(train_X, train_Y)
    mean_L2.append((np.mean((regr.predict(cross_val_X) - cross_val_Y) ** 2)))
    # Explained variance score: 1 is perfect prediction
    variance_L2.append(regr.score(cross_val_X, cross_val_Y))

    # Logistic Regression-L1
    regr = linear_model.LogisticRegression(penalty='l1')
    regr.fit(train_X, train_Y)
    mean_L1.append(np.mean((regr.predict(cross_val_X) - cross_val_Y) ** 2))
    # Explained variance score: 1 is perfect prediction
    variance_L1.append(regr.score(cross_val_X, cross_val_Y))

avg_LinearRegression=np.mean(mean_LinearRegression)
avg_L2=np.mean(mean_L2)
avg_L1=np.mean(mean_L1)
avg_DT=np.mean(mean_DT)
avg_RF=np.mean(mean_RF)
avg_MLP=np.mean(mean_MLP)

print (' Mean Error (Linear Regression)\n',avg_LinearRegression)
print (' Mean Error (Logistic Regression - L2)\n',avg_L2)
print (' Mean Error (Logistic Regression - L1)\n',avg_L1)
print (' Mean Error (Decision Tree)\n',avg_DT)
print (' Mean Error (Random Forest)\n',avg_RF)
print (' Mean Error (Multi Layer Perceptron)\n',avg_MLP)