# ENSEMBLE (pip install ensembles)

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.data_import(data, label_output, encode = 'label', split = True, stratify = True, split_size = 0.1)**

Function is used for providing the input.

__Parameters__ :
> * **data** : Pandas DataFrame
>> The dataset needs to be passed as the parameter, the dataset should not contain any missing values.
> * **label_output** : string
>> The column name which contains the output needs to be passed as the parameter.
> * **encode** : string, optional (default = 'label') {'binary', 'hashing', 'backward_difference', 'helmert', 'sum', 'polynomial'}
>> The encoding that needs to be performed on the data. For more information [click here](https://github.com/wdm0006/categorical_encoding)
> * **split** : bool, optional (default = True)
>> Performing a split to obtain test data (True), Test Data not required (False)
> * **stratify** : bool, optional (default = True)
>> Stratified split (True) or random split (False)
> * **split_size** : float, optional (default = 0.1)
>> The split ratio for training and testing.

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.parameter_set_gradient_boosting(hyper_parameter_optimisation = False, eval_metric = None, booster = ['gbtree'], silent = [0], eta = [0.3], gamma = [0], max_depth = [6], min_child_weight = [1], max_delta_step = [0], subsample = [1], colsample_bytree = [1], colsample_bylevel = [1], lambda_xgb = [1], alpha = [0], tree_method = ['auto'], sketch_eps = [0.03], scale_pos_weight = [0], lambda_bias = [0], objective =['reg:linear'], base_score = [0.5])**

Setting the parameters for gradient boosting. The parameter values have to be entered in the list form, whether single (hyper_parameter_optimisation = False) or multiple values (hyper_parameter_optimisation = True) are used

__Parameters__ :
> * **hyper_parameter_optimisation** : bool, optional (default = False)
>> if False, hyper parameter optimisation will not be performed. <br /> if True, hyper parameter optimisation will be performed across the search space, that is the multiple values entered for the parameters of the gradient boosting model. Using [hyperopt](https://github.com/hyperopt/hyperopt) 
>* **Documentation of the Gradient Boosting model (XGBoost) can be obtained [here](https://github.com/dmlc/xgboost/blob/master/doc/parameter.md)**

__Returns__ : 
>* **Dictionary containg the respective parameter names and values.**

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.parameter_set_decision_tree(criterion = ['gini'], splitter = ['best'], max_depth = [None], min_samples_split = [2], min_samples_leaf = [1], min_weight_fraction_leaf = [0.0], max_features = [None], random_state = [None], max_leaf_nodes = [None], class_weight = [None], presort = [False])**

Setting the parameters for the decision tree classifier. The parameter values have to be entered in the list form, whether single or multiple values are used. Hyper parameter optimisation will be perfromed when multiple values are entered for the parameters. Using [gridsearch](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#examples-using-sklearn-grid-search-gridsearchcv)

__Parameters__ :
>* **Documentation of the Decision Tree model can be obtained [here](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)**

__Returns__ : 
>* **Dictionary containg the respective parameter names and values.**

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.parameter_set_random_forest(n_estimators = [10], criterion = ['gini'], max_depth = [None], min_samples_split = [2], min_samples_leaf = [1], min_weight_fraction_leaf = [0.0], max_features = ['auto'], max_leaf_nodes = [None], bootstrap = [True], oob_score = [False], random_state = [None], verbose = [0],warm_start = [False], class_weight = [None])**

Setting the parameters for the random forest classifier. The parameter values have to be entered in the list form, whether single or multiple values are used. Hyper parameter optimisation will be perfromed when multiple values are entered for the parameters. Using [gridsearch](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#examples-using-sklearn-grid-search-gridsearchcv)

__Parameters__ :
>* **Documentation of the Random Forest model can be obtained [here](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)**

__Returns__ : 
>* **Dictionary containg the respective parameter names and values.**

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.parameter_set_linear_regression(fit_intercept = [True], normalize = [False], copy_X = [True])**

Setting the parameters for the linear regression model. The parameter values have to be entered in the list form, whether single or multiple values are used. Hyper parameter optimisation will be perfromed when multiple values are entered for the parameters. Using [gridsearch](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#examples-using-sklearn-grid-search-gridsearchcv)

__Parameters__ :
>* **Documentation of the Linear Regression model can be obtained [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)**

__Returns__ : 
>* **Dictionary containg the respective parameter names and values.**

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.parameter_set_logistic_regression(penalty = ['l2'], dual = [False], tol = [0.0001], C = [1.0], fit_intercept = [True], intercept_scaling = [1], class_weight = [None], random_state = [None], solver = ['liblinear'], max_iter = [100], multi_class = ['ovr'], verbose = [0], warm_start = [False])**

Setting the parameters for the logistic regression model. The parameter values have to be entered in the list form, whether single or multiple values are used. Hyper parameter optimisation will be perfromed when multiple values are entered for the parameters. Using [gridsearch](http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#examples-using-sklearn-grid-search-gridsearchcv)

__Parameters__ :
>* **Documentation of the Logistic Regression model can be obtained [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)**

__Returns__ : 
>* **Dictionary containg the respective parameter names and values.**

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.train_base_models(model_list, parameters_list, save_models = False)**

The function trains the base models that the user has passed as a parameter parallely using joblib.

__Parameters__ :
>* **model_list** : list, optional {'gradient_boosting', 'decision_tree', 'random_forest', 'linear_regression', 'logistic_regression'}
>> The list can contain any number of models, and models can be repeated.
>* **parameters_list** : dictionary
>> The parameters of the respective models have to be entered in the same order as how the names of its models have been entered in the model_list parameter. The parameters are dictionaries that are returned on calling the respective parameter set functions as described before.
>* **save_models** : bool, optional (default = False)
>> If True, all the base models will be saved in pkl files using joblib. To get the base models and perform further operations using them call the get_base_models() function (ensemble.get_base_models()), the function will return all the base model objects.

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.assign_weights(weights = 'default', hyper_parameter_optimisation = False)**

The function needs to be called if weighted average is going to be performed, to set the weights for performing weighted average.

__Parameters__ :
>* **weights** : list, optional (default = 'default')
>> The value 'default' will assign default weights, that is range (1-10) (if hyper_parameter_optimisation = True) otherwise equal weights = 1 (if hyper_parameter_optimisation = False)<br /> The other option is to manually pass the weights that need to be assigned to the model, or multiple weights for each model (nested list can be passed) can be passed and it will be optimised using [hyperopt](https://github.com/hyperopt/hyperopt) 
>* **hyper_parameter_optimisation** : bool, optional (default = False)
>> Needs to be True when multiple weights are passed for a model (nested list). False when weights are manually assigned or when using equal weights.

__Returns__ : 
>* **List containing the weights that will be used for performing the weighted average.**


------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.train_ensemble_models(stack_model_list = [], stack_parameters_list = [], blend_model_list = [], blend_parameters_list = [], perform_weighted_average = None, weights_list = None, save_models = False)**

The function needs to be called for training the ensemble models.

__Parameters__ :
>* **stack_model_list** : list or empty, optional {'gradient_boosting', 'decision_tree', 'random_forest', 'linear_regression', 'logistic_regression'}
>> The list can contain any number of models, and models can be repeated. When not performing stacking, an empty list will be passed
>* **stack_parameters_list** : dictionary
>> The parameters of the respective models have to be entered in the same order as how the names of its models have been entered in the stack_model_list parameter. The parameters are dictionaries that are returned on calling the respective parameter set functions as described before.
>* **blend_model_list** : list or empty, optional {'gradient_boosting', 'decision_tree', 'random_forest', 'linear_regression', 'logistic_regression'}
>> The list can contain any number of models, and models can be repeated. When not performing blending, an empty list will be passed
>* **blend_parameters_list** : dictionary
>> The parameters of the respective models have to be entered in the same order as how the names of its models have been entered in the blend_model_list parameter. The parameters are dictionaries that are returned on calling the respective parameter set functions as described before.
>* **perform_weighted_average** : bool or None, optional (defualt = None)
>> To specify wether to perform weighted average of the base models or not.
>* **weights_list** : list
>> The list that is returned by the ensemble.assign_weights() function

------------------------------------------------------------------------------------------------------------------------------------------------
**ensemble.test_data()**

The function needs to be called for measuring the performance of the model on the test dataset.

------------------------------------------------------------------------------------------------------------------------------------------------

