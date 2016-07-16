importers when locating support scripts as well as when importing modules.
import sys
import importlib.machinery # importlib first so we can test #15386 via -m
import importlib.util
import types
from pkgutil import read_code, get_importer
# TODO: Replace these helpers with importlib._bootstrap_external functions.
spec = importlib.util.find_spec(mod_name)
# importlib, where the latter raises other errors for cases where
"""Execute a module's code without importing it
importer = get_importer(path_name)
# Trying to avoid importing imp so as to not consume the deprecation warning.
if type(importer).__module__ == 'imp':
if type(importer).__name__ == 'NullImporter':
if isinstance(importer, type(None)) or is_NullImporter:
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

 #Performing cross validation.
skf=StratifiedKFold(train_Y, n_folds=2)
for train_index, cross_val_index in skf:
    
    mlp_train_X, mlp_cross_val_X = train_X.iloc[train_index],train_X.iloc[cross_val_index]
    mlp_train_Y, mlp_cross_val_Y = train_Y.iloc[train_index],train_Y.iloc[cross_val_index]
    mlp_train_X = mlp_train_X.as_matrix()
    mlp_train_Y = mlp_train_Y.as_matrix()
    
    

def keras_fmin_fnct(space):

    
    print('a')
    param = multi_layer_perceptron_parameters()
    dim_layer = param[0]
    activation_layer_1 = param[1]
    init_layer_1 = param[2]
    activation_layer_2 = param[3]
    optimizer = param[4]
    print('b')
    mlp_train_X,mlp_train_Y, mlp_cross_val_X, mlp_cross_val_Y = data()
    
    model = Sequential()
    model.add(Dense(output_dim = dim_layer, input_dim = train_X.shape[1], init = init_layer_1, activation = activation_layer_1))
    model.add(Dense(output_dim = 1, input_dim = dim_layer,activation = activation_layer_2))
    model.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics = ['accuracy'])
    model.fit(mlp_train_X, mlp_train_Y, nb_epoch = 5, batch_size = 128)
    auc,predict = cross_val_multi_layer_perceptron(mlp_cross_val_X,mlp_cross_val_Y)
    
    return {'loss': 1-auc, 'status': STATUS_OK, 'model': model}


def get_space():
    return {
    }
