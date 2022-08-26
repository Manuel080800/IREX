## -*- coding: utf-8 -*-

"""
    IREX: A reusable process for the iterative refinementand explanation
    of classification models.

    @Authors:
        1 .- Manuel A. Cetina Aguilar.
        2 .- Cristian E. Sosa Espadas.
        3 .- Jose A. Soladrero Gonzalez.
        4 .- Jesús M. Dárias.

    Script developed using Spyder.
"""

def loadImports():
    """
    Import the necessary modules from the script.

    Parameters:
        None.

    Exceptions:
        None.

    Returns:
        None.
    """

    def imports(moduleName, shortName = None, asAttribute = False):
        """
        Import the modules globally in the script.

        Imports all the necessary modules globally for the
        operation of the script.

        Parameters:
            moduleName  -- Name of the module to import.
            shortName   -- Short name of the imported module.
            asAttribute -- Boolean that determines if is an
                           attribute or function.

        Exceptions:
            KeyError -- Name or short name of the module
                        do not exist.

        Returns:
            None.
        """

        import importlib

        ## If the short name is not defined.
        if shortName is None: 
            ## Then it takes the name of the module.
            shortName = moduleName

        ## If it is not an attribute or function of the module.
        if asAttribute is False:
            ## Then fully import the module.
            globals()[shortName] = importlib.import_module(moduleName)
        else:
            ## If not, then import the attribute or function of the module.
            globals()[shortName] = getattr(importlib.import_module(moduleName),
                                         shortName)

    ## Import of all the necessary libraries.
    imports('joblib')
    imports('dice_ml')
    imports('shap')
    imports('lime')
    imports('seaborn', 'sns')
    imports('pandas', 'pd')
    imports('matplotlib.pyplot', 'plt')
    imports('numpy', 'np')
    imports('operator', 'itemgetter', True)
    imports('sklearn.model_selection', 'train_test_split', True)
    imports('sklearn.model_selection', 'cross_val_score', True)
    imports('explainerdashboard', 'ClassifierExplainer', True)
    imports('sklearn.metrics', 'ConfusionMatrixDisplay', True)
    imports('sklearn.metrics', 'precision_recall_fscore_support', True)
    imports('sklearn.metrics', 'classification_report', True)
    imports('sklearn.metrics', 'confusion_matrix', True)
    imports('sklearn.metrics', 'accuracy_score', True)
    imports('sklearn.metrics', 'mean_squared_error', True)
    imports('sklearn.neural_network', 'MLPClassifier', True)
    imports('imblearn.over_sampling', 'SMOTE', True)
    imports('alibi.explainers.ale', 'ALE', True)
    imports('alibi.explainers.ale', 'plot_ale', True)

def loadModel():
    """
    Generation of the necessary model for the explanation.

    A dictionary will be generated with the necessary model
    for the explanation.

    Parameters:
        None.

    Exceptions:
        None.

    Returns:
        model -- Dictionary with the model defined
                 for the explanation.
    """

    ## The dictionary for the model is generated.
    model = dict()

    ## The dataset is defined with the model to explain.
    model['path_dataset'] = "Dataset_MO_ENG.csv"

    return model

def loadExpectedOutputs():
    """
    Generation of the expected answers for the explanation.

    A dictionary will be generated with the expected answers
    necessary for the explanation to work.

    Parameters:
        None.

    Exceptions:
        None.

    Returns:
        expectedOutputs -- Dictionary with the expected
                           answers for the explanation.
    """

    ## The dictionary for the expected answers is generated.
    expectedOutputs = dict()

    ## The dataset with the expected responses is defined.
    expectedOutputs['path_dataset_qa'] = "Dataset_QA_ENG.csv"

    return expectedOutputs

def configParameters():
    """
    Generation of the necessary configurations for
    the explanation.

    A dictionary is generated with the configuration
    parameters for the explanation job.

    Parameters:
        None.

    Exceptions:
        None.

    Returns:
        parameters -- Dictionary with settings for
        explanation.
    """

    ## The dictionary is generated with the explanation settings
    parameters = dict()

    ## The column, shape and grouping name of the dataset to be
    ## explained are defined.
    parameters['grouping_name'] = 'Target' # *
    parameters['class_dic'] = {1: 0 , 2: 0, 3:1, 4:2, 5:2}
    ## * If this data is None or empty string then class_dic
    ##   will be ignored

    ## The class names present in the dataset to be explained are defined.
    parameters['target_names'] = ["Low", "Medium", "High"] # *
    ## * If grouping_name is None or empty, then a class name must exist.
    ##   If not, then the total class name must agree with the total
    ##   groupings defined in class_dic.

    ## The class names present in the dataset to be plotted are defined.
    parameters['name_class'] = ["Class Low", "Class Medium", "Class High"] # *
    parameters['list_color'] = ['gray', 'black', 'red']
    ## * If grouping_name is None or empty, then a class name must exist.
    ##   If not, then the total class name must agree with the total
    ##   groupings defined in class_dic.

    ## The questions to exclude from the dataset are defined.
    parameters['unrelated_questions'] = [102, -1] # *
    ## * If this data is an empty list then this data will be ignored.

    ## The list of questions eliminated at the start of the first
    ## iteration is defined.
    parameters['list_contr'] = [3, 4, 5, 14, 21, 24, 25, 26, 27, 29, 30, 32,
                                48, 49, 51, 54, 55, 58, 59, 60, 62, 63, 66,
                                68, 70, 75, 78, 83, 85, 86, 87, 89, 90, 91,
                                94, 95, 96, 98, 100, 101]

    ## The regular and alpha parameters of the training are defined.
    parameters['regul_param_normal'] = 10.0 ** - np.arange(-2, 7)
    parameters['regul_param_alpha'] = 10.0 ** - np.arange(0, 7)

    ## The alpha, random state and learning rate training parameters
    ## are defined for each iteration.
    parameters['learning_rate_init'] = 0.1
    parameters['random_state'] = 42
    parameters['alpha'] = 0.001

    ## Positive and negative threshold values are defined for the
    ## slope qualification process.
    parameters['positive_threshold'] = 0.01
    parameters['negative_threshold'] = -0.01

    ## The qualification mode of each answer obtained is defined.
    parameters['selection_mode'] = 1 # *
    ## * The following modes are available for the qualification
    ##   the answers present in the dataset:
    ##
    ##     1. Selection of a single class.
    ##     2. Selection of multiple classes with OR type condition.
    ##     3. Selection of multiple classes with AND type condition.
    ##     4. Selection of multiple classes with AND type condition
    ##        when they are Out-of-threshold range.

    ## The position of the class to qualify is defined.
    parameters['select_class_mode'] = 2 # *
    ## * Position is related to the number of classes
    ##   present in target_names and is required only when
    ##   selection_mode is set to 1.

    ## The positions of the classes to qualify are defined.
    parameters['selection_class'] = [0, 2] # *
    ## * Positions are related to the number of classes
    ##   present in target_names and is required only when
    ##   selection_mode is set to 2, 3 and 4.

    ## The map of expected responses is defined.
    parameters['expected_response_map'] = [[0, 1, 0, 1],
                                           [1, 0, 1, 0]] # *
    ## * A two-dimensional list is defined, where each element of
    ##   is a list of 4 numerical elements limited to 0 and 1. The
    ##   number of elements of this is given by the number of
    ##   classes to qualify.
    ## 
    ##   The 4 numerical elements must meet the following conditions:
    ##
    ##     Element 1. Score is 1 and expected response is 0.
    ##     Element 2. Score is 1 and expected response is 1.
    ##     Element 3. Score is 0 and expected response is 1.
    ##     Element 4. Score is 0 and expected response is 0.

    return parameters

def loadData():
    """
    Generation of the necessary data for the explanation.

    A dictionary will be generated with all the necessary
    data for the explanation.

    Parameters:
        None.

    Exceptions:
        None.

    Returns:
        data -- Dictionary with all the data defined for
        the explanation.
    """

    ## The dictionary is generated with the data.
    data = dict()

    ## The datasets for storing the auxiliary data are defined.
    data['path_dataset_qs'] = "Dataset_QS_ENG.csv"
    data['path_dataset_qd']= "Dataset_QD_ENG.csv"

    ## The oversample function used is defined.
    data['random_state'] = 13
    data['oversample'] = SMOTE(random_state = data['random_state'])

    ## The seed used for training the neural network is defined.
    data['seed'] = 1

    ## The number of iterations performed in the explanation is defined.
    data['iteration'] = 0

    ## The names of the explanation methods used are defined.
    data['name_methods'] = ["Importance", "LIME", "SHAP", "ALE"]

    return data

def explain(data, model, config, expertKnowledge):
    """
    Explanation of the neural network and improvement of this.

    Explanation of the neural network from the iterative
    improvement process by retraining the network by
    qualifying the answers and using explanation methods
    such as Important Variable, ALE, LIME and SHAP.

    Parameters:
        data            -- Dictionary with explanation operating data.
        model           -- Dictionary with pattern for explanation.
        config          -- Dictionary with the necessary configurations
                           to be able to carry out the explanation.
        expertKnowledge -- Short name of the importDictionary with the
                           expected answers for the explanation.ed module.

    Exceptions:
        None.

    Returns:
        None.
    """

    ## Loading data from the dataset.
    df = pd.read_csv(model ['path_dataset'])
    
    ## Elimination of unrelated questions from the dataset.
    if len(config['unrelated_questions']) != 0:
        df = df.drop(df.columns[config['unrelated_questions'][0]:
                                config['unrelated_questions'][1]],
                     axis=1)

    ## Grouping of the target according to the defined dictionary.
    if config['grouping_name'] != None and config['grouping_name'] != '':
        df[config['grouping_name']] = df[config['grouping_name']
                                         ].map(config['class_dic'])

    ## Assignment of local variables according to the data necessary
    ## for the neural network.
    train_cols = df.columns [0:-1]
    label = df.columns [-1]
    X = df [train_cols]
    y = df [label]

    ## Generation of the question status list automatically.
    question = list(range(1, df.shape[1]))
    status = ['In use' for _ in range(1, df.shape[1])]
    ds = pd.DataFrame({'Question': question, 'Status': status})

    ## Save the changed statuses in the dataset.
    ds.to_csv(data['path_dataset_qs'], index = False)

    ## Generation of the data structure by iteration automatically.
    columns = ['Question', 'Acurracy global']
    
    for index in range(0, len(config['name_class'])):
        columns.append('Precision_' + str(index))
        columns.append('Recall_' + str(index))
        columns.append('F1_score_' + str(index))
        columns.append('Support_' + str(index))
        
    dd = pd.DataFrame(columns = columns)

    ## Save the changed data in the dataset.
    dd.to_csv(data['path_dataset_qd'], index = False)

    ## The prepared data of the dataset is printed.
    print("Dataset used for the explanation.")
    print(df)
    print()

    ## The number of elements present in the grouped target is printed.
    print("Number of elements grouped by class.")
    print(y.value_counts())
    print()

    ## Application of the oversample to the prepared data.
    print("Oversample application.")
    X, y = data['oversample'].fit_resample(X, y)
    print()

    ## The results obtained after applying the oversample are printed.
    print("Number of elements grouped by class.")
    print(y.value_counts())
    print()

    ## Preparation of the parameters of the neural network.

    ## We proceed to define the parameters necessary for the training
    ## of the neural network.
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33,
                                                  random_state=data['seed'])

    ## Neural network training.

    ## The alpha parameters and the initial learning rate are adjusted
    ## to obtain the best performance with the model using cross validation.

    ## Local variables are defined for the storage of the scores obtained
    ## by the training.
    
    print("Neural network training.")
    cv_scores_mean = []
    cv_scores_std = []
    regul_param_range = config['regul_param_normal']

    ## Training and validation of different configurations.
    for regul_param in regul_param_range:

        ## Increase the max_iter parameter until it converges.
        mlp = MLPClassifier(hidden_layer_sizes = (10,),
                            activation = 'logistic', solver = 'adam',
                            alpha = regul_param, learning_rate = 'constant',
                            learning_rate_init = 0.0001, max_iter = 100000,
                            random_state = data['seed'])

        scores = cross_val_score(mlp, X, y, cv = 5, scoring = 'f1_macro')

        cv_scores_mean.append(scores.mean())
        cv_scores_std.append(scores.std())

    ## The results obtained during the training are printed.
    print(cv_scores_mean, cv_scores_std)
    print()

    ## We proceed to draw the learning curve graph according to the data
    ## obtained by the training.
    print("Generation of the learning curve graph.")
    plt.figure()

    ## The average accuracy line is drawn on the test parts.
    plt.plot(np.log10(regul_param_range), cv_scores_mean,
             color = "g", label = "Test")

    ## The standard deviation band is drawn.
    lower_limit = np.array(cv_scores_mean) - np.array(cv_scores_std)
    upper_limit = np.array(cv_scores_mean) + np.array(cv_scores_std)
    plt.fill_between(np.log10(regul_param_range), lower_limit, upper_limit,
                     color = "#DDDDDD")

    ## Generating the graph.
    plt.title("Learning curve")
    plt.xlabel("Alpha 10^{X}"), plt.ylabel("F1"),
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("Learning curve Alpha 10^{X} - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## The training is performed again keeping the same parameters
    ## except for the alpha value that changes to 1.

    ## Local variables are defined for the storage of the scores
    ## obtained by the training.

    print("Neural network training.")
    cv_scores_mean = []
    cv_scores_std = []
    regul_param_range = config['regul_param_alpha']

    ## Training and validation of different configurations.
    for regul_param in regul_param_range:

        ## Increase the max_iter parameter until it converges.
        mlp = MLPClassifier(hidden_layer_sizes = (10,),
                            activation = 'logistic', solver = 'adam',
                            alpha = 1, learning_rate = 'constant',
                            learning_rate_init = regul_param,
                            max_iter = 100000, random_state = data['seed'])

        scores = cross_val_score(mlp, X, y, cv = 5, scoring = 'f1_macro')

        cv_scores_mean.append(scores.mean())
        cv_scores_std.append(scores.std())

    ## The results obtained during the training are printed.
    print(cv_scores_mean, cv_scores_std)
    print()

    ## We proceed to draw the learning curve graph according to the data
    ## obtained by the training.
    print("Generation of the learning curve graph.")
    plt.figure()

    ## The average accuracy line is drawn on the test parts.
    plt.plot(np.log10(regul_param_range), cv_scores_mean,
             color = "g", label = "Test")

    ## The standard deviation band is drawn.
    lower_limit = np.array(cv_scores_mean) - np.array(cv_scores_std)
    upper_limit = np.array(cv_scores_mean) + np.array(cv_scores_std)
    plt.fill_between(np.log10(regul_param_range), lower_limit, upper_limit,
                     color = "#DDDDDD")

    ## Generate the graph.
    plt.title("Learning curve")
    plt.xlabel("Learning Rate 10^{-X}"), plt.ylabel("F1"),
    plt.legend(loc = "best")
    plt.tight_layout()
    plt.savefig("Learning curve Rate 10^{-X} - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Generation of the neural network model.

    ## The parameters of the final model are modified according to the
    ## data obtained in the training of the neural network.

    ## The value of the alpha parameter is kept at 1 as well as the value
    ## of the seed of random_state at 77.
    print("Generation of the neural network.")
    mlp = MLPClassifier(hidden_layer_sizes = (10,), activation = 'logistic',
                        solver = 'adam', alpha = 1, learning_rate = 'constant',
                        learning_rate_init = 0.0001, max_iter = 100000,
                        random_state = 77)

    ## We print the data of the final generated model.
    print(mlp.fit(X_train, y_train))
    print()

    ## We safeguard the final generated model
    print("Neural Network Safeguard.")
    joblib.dump(mlp,"model_depression.pkl")
    print("Save model_depression.pkl")
    print()

    ## Generation of the confusion matrix of the model.

    ## We obtain the statistics of the final model to generate its
    ## confusion matrix.

    ## The data for the confusion matrix's generation is defined.
    print("Generation of the confusion matrix.")
    confusion_matrix = ConfusionMatrixDisplay.from_estimator(mlp, X_test,
                       y_test, display_labels = config['target_names'],
                       cmap = plt.cm.Blues)

    confusion_matrix.ax_.set_title("Confusion Matrix")
    plt.savefig("Confusion Matrix - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## We get the detailed statistics of the final model.

    ## The necessary detailed prediction is generated.
    y_pred = mlp.predict(X_test)

    ## The detailed statistics of the model are printed.
    print("Obtaining the statistics of the model.")
    print('Classification accuracy =',
          accuracy_score(y_test,y_pred) * 100,'%\n')
    print(classification_report(y_test,y_pred))
    print()

    # The data used for the test of the neural network model is shown.
    print("Model test data.")
    print(X_test)
    print()

    ## Explanation of the model using ALE.

    ## In analysis of the model through the results obtained by ALE,
    ## it is possible to identify the behavior of the neural network.

    ## The necessary parameters for the use of ALE are established
    ## according to the model.

    proba_fun_lr = mlp.predict_proba
    proba_ale_lr = ALE(proba_fun_lr, feature_names = train_cols,
                       target_names = config['target_names'])
    proba_exp_lr = proba_ale_lr.explain(X_train.to_numpy())

    # The graphs of all the data present in the dataset used are shown.
    print("Generation of ALE graphs.")
    fig, ax = plt.subplots()
    plot_ale(proba_exp_lr, n_cols=2, features=list(range(df.shape[1]-1)),
             ax=ax, fig_kw={'figwidth': 10, 'figheight': 180})
    plt.savefig("ALE graphs - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Explanation of the model using LIME.

    ## We proceed to explain the improved model using LIME for all classes.

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
                                        categorical_features = train_cols,
                                        feature_names=train_cols,
                                        class_names=config['target_names'],
                                        discretize_continuous=True)

    ## The first, second and last individual present in the test
    ## data is explained.
    single_relationship = ['first', 'second', 'latest']
    map_relationship = [0,1,-1]
    for individual_index in map_relationship:
        print("Generation of the LIME explanation graph of the " +
              single_relationship[map_relationship.index(individual_index)] +
              " individual")
        exp = explainer.explain_instance(X_test.to_numpy()[individual_index],
                                         mlp.predict_proba, num_features=6,
                                         top_labels=1)

        for i in range(0, len(config['target_names'])):
            try:
                exp.as_pyplot_figure(label=i)
                plt.savefig("LIME graphs of the " + 
                            single_relationship[map_relationship.index(
                                individual_index)] + " - initial.jpg",
                            bbox_inches='tight')
                plt.show()
                break
            except:
                pass
        print()

    ## Explanation of the model using SHAP.

    ## SHAP explanatory values are displayed according to the
    ## class to which the selected individual belongs according to
    ## the prediction of the model.

    ## The first, second and last individual present in the test
    ## data is explained.
    single_relationship = ['first', 'second', 'latest']
    map_relationship = [0,1,-1]
    for individual_index in map_relationship:
        print("Generation of the SHAP explanation graph of the " +
              single_relationship[map_relationship.index(individual_index)] +
              " individual")
        individual = X_test.iloc[[individual_index]]

        ## Individual explanation by SHAP.
        explainer = shap.KernelExplainer(mlp.predict_proba, X_train,
                                         feature_names = train_cols,
                                         output_names = config['target_names'])
        shap_values = explainer.shap_values(individual)
        clase = mlp.predict(individual)[0]

        fig = plt.gcf()
        shap.summary_plot(shap_values[clase], individual,
                          feature_names = train_cols, plot_type = "bar")
        fig.savefig("SHAP graphs of the " + 
                    single_relationship[map_relationship.index(
                    individual_index)] + " - Initial.jpg",
                    bbox_inches='tight')
        print()

    ## Explanation using heatmaps.

    ## The importance of the data present in the neural network is shown
    ## from the explanation methods.

    ## Items currently in use are retrieved.
    ds_item = ds[ds.Status == "In use"].reset_index(drop=True)

    ## The data is prepared for the generation of heatmaps.
    column_Map = ds_item['Question'].tolist()
    content_Map = [False for x in range(1, df.shape[1])]

    ## The next Items to be deleted will be marked.
    for index in range(0, len(content_Map)):
        if data['iteration'] == 0:
            for delete in config['list_contr']:
                if column_Map[index] == delete:
                    content_Map [index] = True
        else:
            for delete in question_anomaly:
                if column_Map[index] == delete:
                    content_Map [index] = True

    ## The dataframe is created with the prepared data.
    df_mask = pd.DataFrame([content_Map], columns = column_Map)
    df_mask_imp = pd.concat([df_mask, df_mask, df_mask],
                            ignore_index = True, axis = 0)
    df_mask_all = pd.concat([df_mask, df_mask, df_mask, df_mask],
                            ignore_index = True, axis = 0)

    ## The heatmap mask dataset is printed.
    print("Heatmap mask dataset.")
    print(df_mask)
    print()

    ## Identification of features variables.

    ## Identification of features from the randomness of the
    ## answers as questions of the data.

    ## Loading the data for explainability.
    explainer = ClassifierExplainer(mlp, X_test, y_test)

    ## Class index local variable.
    map_class = []

    for iterable_class in range(0, len(config['name_class'])):
        ## Explainability of the data by means of the heat-
        ## map of all the classes.
        print(config['name_class'][iterable_class] +
              " Feature Important heatmap generation.")
        df_importance = explainer.permutation_importances(iterable_class
                                                          ).sort_index()
        df_importance.index = column_Map
        df_importance.drop("Score",inplace=True, axis=1)
        df_importance.drop("Feature",inplace=True, axis=1)
        df_importance.loc[df_importance['Importance'] < 0, 'Importance'] = 0
        map_class.append(df_importance)

        ## Generation of the heatmap.
        plt.figure(figsize = (28,6))
        sns.heatmap(map_class[iterable_class].transpose(),
                    cmap="Reds", cbar = False)
        sns.heatmap(map_class[iterable_class].transpose(),
                    cmap="Blues", yticklabels = True, xticklabels = True,
                    mask = df_mask.to_numpy())
        plt.savefig(config['name_class'][iterable_class] +
              " Important Variable heatmap - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

    ## Merging of all the classes' heatmaps into one.

    ## Fusion of the previous heatmaps.
    df_importance = pd.concat(map_class, axis=1)
    df_importance.columns = config['name_class']

    ## Generation of the heatmap.
    print("Generation of the heatmap of Important Variable of all classes.")
    plt.figure(figsize = (32,6))
    sns.heatmap(df_importance.transpose(), cmap = "Reds", cbar = False)
    sns.heatmap(df_importance.transpose(), cmap = "Blues", yticklabels = True,
                xticklabels = True ,mask = df_mask_imp.to_numpy())
    plt.savefig("All classes of Important Variable heatmap - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Identification of feature variables using SHAP.

    # Contribution in the final probability of each question for specific
    ## instances at the local level. For each class, a map is first shown
    ## with all the individuals and the contribution of each question,
    ## and then a map made with the mean of these values in absolute value.
    ## This seeks to globalize the scope of SHAP.

    ## Class index local variable.
    map_class = []

    ## Explainability of the data by means of the heatmap of all the classes.

    ## Definition of explainability data.

    for iterable_class in range(0, len(config['name_class'])):
        df_shap = explainer.get_shap_values_df(iterable_class)
        df_shap.columns = column_Map

        ## Generation of the heatmap.
        print(config['name_class'][iterable_class] +
              " SHAP heatmap generation.")
        plt.figure(figsize = (28,14))
        sns.heatmap(df_shap, cmap="vlag_r", yticklabels=True,
                    xticklabels=True, center = 0)
        plt.savefig(config['name_class'][iterable_class] +
              " SHAP heatmap - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Fusion of the data used.

        df_shap_mean = df_shap.abs().mean(axis=0).to_frame()
        df_shap_mean.columns= ["Mean SHAP"]
        map_class.append(df_shap_mean)

        ## Generation of the heatmap.
        print(config['name_class'][iterable_class] +
              " SHAP average heatmap generation.")
        plt.figure(figsize = (28,6))
        sns.heatmap(map_class[iterable_class].transpose(), cmap="Reds",
                    cbar = False)
        sns.heatmap(map_class[iterable_class].transpose(), cmap="Blues",
                    yticklabels = True, xticklabels = True,
                    mask = df_mask.to_numpy())
        plt.savefig(config['name_class'][iterable_class] +
              " SHAP average heatmap - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

    ## Merging of all the class heatmaps into one map.

    ## Fusion of the previous heatmaps.
    df_shap = pd.concat(map_class, axis=1)
    df_shap.columns = config['name_class']

    ## Generation of the heatmap.
    print("Generation of the heatmap of SHAP of all classes.")
    plt.figure(figsize = (32,6))
    sns.heatmap(df_shap.transpose(), cmap="Reds", cbar = False)
    sns.heatmap(df_shap.transpose(), cmap="Blues", yticklabels = True,
                xticklabels = True, mask = df_mask_imp.to_numpy())
    plt.savefig("All classes of SHAP heatmap - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Identification of feature variables using LIME.

    ## Contribution in the final probability of each question for
    ## specific instances at the local level. For each class, a map
    ## is first shown with all the individuals and the contribution
    ## of each question, and then a map made with the mean of these values
    ## in absolute value. This seeks to globalize the scope of LIME.

    ## Data preparation for explainability.
    lime_exp = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
            categorical_features = train_cols, feature_names=train_cols,
            class_names=config['target_names'], discretize_continuous=True)

    ## Class index local variable.
    map_class = []

    ## Obtaining the data for the generation of the explanations by LIME
    ## about all the classes.

    ## Local variables for LIME array data.
    exp_matrix = [[] for _ in range(len(config['name_class']))]

    ## Recovering the data needed for LIME.
    for x in X_test.to_numpy():

        ## Auxiliary temporary variables.
        exp_list = [[] for _ in range(len(config['name_class']))]

        ## Loading the data for explainability.
        exp = lime_exp.explain_instance(x, mlp.predict_proba,
                                        num_features = df.shape[1] - 1,
                                        top_labels = len(config['name_class']))

        ## Recovering the data.
        for elements in range(0, len(config['name_class'])):
            temp = exp.as_map()[elements]
            temp.sort(key=itemgetter(0))

            ## Saving data of all the classes.
            for tup in temp:
                exp_list[elements].append(tup[1])

            exp_matrix[elements].append(exp_list[elements])

    ## Explainability of the data by means of the heatmap of
    ## all the classes.

    # Generation of the heatmap.

    for iterable_class in range(0, len(config['name_class'])):
        print(config['name_class'][iterable_class] +
              " LIME heatmap generation.")
        plt.figure(figsize = (28,14))
        lime_df = pd.DataFrame(exp_matrix[iterable_class])
        lime_df.columns = column_Map
        sns.heatmap(lime_df, cmap="vlag_r",yticklabels=True, xticklabels=True)
        plt.savefig(config['name_class'][iterable_class] +
              " LIME heatmap - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Fusion of the data used.

        lime_df_mean = lime_df.abs().mean(axis = 0).to_frame()
        lime_df_mean.columns= ["Mean LIME"]
        map_class.append(lime_df_mean)

        print(config['name_class'][iterable_class] +
              " LIME average heatmap generation.")
        plt.figure(figsize = (32,6))
        sns.heatmap(map_class[iterable_class].transpose(), cmap="Reds",
                    cbar = False)
        sns.heatmap(map_class[iterable_class].transpose(), cmap="Blues",
                    yticklabels = True, xticklabels = True,
                    mask = df_mask.to_numpy())
        plt.savefig(config['name_class'][iterable_class] +
              " LIME average heatmap - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

    ## Merging of all the classes heatmaps into one map.
    print("Generation of the heatmap of LIME of all classes.")
    df_lime = pd.concat(map_class, axis=1)
    df_lime.columns = config['name_class']
    plt.figure(figsize = (32,6))
    sns.heatmap(df_lime.transpose(), cmap="Reds", cbar = False)
    sns.heatmap(df_lime.transpose(), cmap="Blues", yticklabels = True,
                xticklabels = True, mask = df_mask_imp.to_numpy())
    plt.savefig("All classes of LIME heatmap - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Identification of feature variables using ALE.

    ## It shows the main effect of each question compared to the
    ## probability that the model predicts in each class. This effect is
    ## shown for both responses. Since there are only 2 answers, the map
    ## is the same but with the colors reversed.

    ## Definition of explainability data.
    proba_fun_lr = mlp.predict_proba
    proba_ale_lr = ALE(proba_fun_lr, feature_names=train_cols,
                       target_names = config['target_names'])
    proba_exp_lr = proba_ale_lr.explain(X_train.to_numpy())

    ## Class index local variable.
    map_class = []

    ## Obtaining the data for the generation of the explanations
    ## by ALE about all the classes.

    ## Local variables for ALE array data.
    ale_list = [[] for _ in range(len(config['name_class']))]

    ## Recovering the data needed for ALE.
    for array in proba_exp_lr.ale_values:

        ## Recovering the data.
        for elements in range(0, len(config['name_class'])):

            ## Saving data of all the classes.
            ale_list[elements].append(array[0][elements])

    ## Data processing for the explainability of ALE.

    ## Data processing of all the classes.
    for elements in range(0, len(config['name_class'])):
        ale_df = pd.DataFrame([ale_list [elements]])
        ale_df = pd.concat([ale_df.multiply(-1), ale_df])
        ale_df.index = [config['name_class'] [elements] + " - False",
                        config['name_class'] [elements] + " - True"]
        ale_df.columns = column_Map
        map_class.append(ale_df)

    ## Explainability of the data by means of the heatmap
    ## of all the class.

    ## Generation of the heatmap.

    for iterable_class in range(0, len(config['name_class'])):
        print(config['name_class'][iterable_class] +
              " ALE heatmap generation.")
        plt.figure(figsize = (28,6))
        sns.heatmap(map_class[iterable_class], cmap="vlag_r",
                    yticklabels=True, xticklabels=True)
        plt.savefig(config['name_class'][iterable_class] +
              " ALE heatmap - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

    ## Fusion of the data used.

    ## Application of absolute value to data.
    for index in range(0, len(map_class)):
        map_class [index] = map_class [index][0:1].abs()

    ## Union of the heatmaps of all the class.
    ale_df = pd.concat(map_class)
    ale_df.index = config['name_class']
    ale_df = ale_df.transpose()

    # Generation of the heatmap.

    print("Generation of the heatmap of ALE of all classes.")
    plt.figure(figsize = (28,6))
    sns.heatmap(ale_df.transpose(), cmap="Reds", cbar = False)
    sns.heatmap(ale_df.transpose(), cmap="Blues", yticklabels = True,
                xticklabels = True, mask= df_mask_imp.to_numpy())
    plt.savefig("All classes of ALE heatmap - Initial.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Comparing of Important Variables Methods for Explainability

    ## The global values of each of the methods used are taken,
    ## their maximums are obtained (the minimums are 0) and a
    ## percentage is calculated based on said maximum, which
    ## indicates how relevant the contribution is for each question.

    # ### All class method comparison

    # The highest data of each method used is obtained.

    for iterable_class in range(0, len(config['name_class'])):
        ## The maximum present value is obtained in each explainability
        ## method.
        max_importance = df_importance[config['name_class'][iterable_class]
                                       ].max()
        max_shap = df_shap[config['name_class'][iterable_class]].max()
        max_lime = df_lime[config['name_class'][iterable_class]].max()
        max_ale  = ale_df[config['name_class'][iterable_class]].max()

        # A new heatmap is created containing all the values of each
        ## method used.

        ## The values are calculated to plot the new heatmap.
        data_importance = df_importance[config['name_class'][iterable_class]
                                        ].multiply(100/max_importance)
        data_lime = df_lime[config['name_class'][iterable_class]
                            ].multiply(100/max_lime)
        data_shap = df_shap[config['name_class'][iterable_class]
                            ].multiply(100/max_shap)
        data_ale = ale_df[config['name_class'][iterable_class]
                         ].multiply(100/max_ale)

        df_general = pd.DataFrame([data_importance, data_lime, data_shap,
                                   data_ale])

        ## Method names are assigned.
        df_general.index = data['name_methods']
        df_general

        # Generation of the heatmap.
        print("Generation of the heatmap of all the comparison methods" +
              " of the " + config['name_class'][iterable_class] + ".")
        plt.figure(figsize = (28,8))
        sns.heatmap(df_general, cmap="Reds", cbar = False)
        sns.heatmap(df_general, cmap="Blues", yticklabels = True,
                    xticklabels = True, mask = df_mask_all.to_numpy())
        plt.savefig("Comparison of all the methods of the " + 
                    config['name_class'][iterable_class] + 
                    " - Initial.jpg",
                    bbox_inches='tight')
        plt.show()
        print()

    while True:

        ## Start of the iterative process of improvement of
        ## the neural network.

        ## Preparing the data for the neural network

        ## The data is prepared in order to be processed by
        ## the neural network.

        ## The iteration number is increased.
        data['iteration'] += 1

        ## The current iteration number is printed.
        print("Iteration number " + str(data['iteration']) + ".")
        print()

        ## Loading data from the dataset.
        df = pd.read_csv(model ['path_dataset'])
        
        ## Elimination of anomalous items from the dataset.
        if len(config['unrelated_questions']) != 0:
            df = df.drop(df.columns[config['unrelated_questions'][0]:
                                    config['unrelated_questions'][1]],
                         axis=1)

        i = 0
        if data['iteration'] == 1:
            ## Elimination of list of conflicting items declared
            ## in global variables.
            for x in config['list_contr']:
                df.drop(df.columns[x - (i + 1)], axis = 1, inplace = True)
                i += 1
                ## Status change of the deleted question in the respective
                ## auxiliary dataset.
                ds.loc[x - 1,'Status'] = 'Delete'

        else:
            ## Elimination of the list of anomadic items defined in the
            ## improvement process.
            ds_delete = ds[ds.Status == "Delete"].reset_index(drop=True)
            for x in ds_delete['Question'].tolist():
                df.drop(df.columns[x - (i + 1)], axis = 1, inplace = True)
                i += 1

        ## Grouping of the target according to the defined dictionary.
        if config['grouping_name'] != None and config['grouping_name'] != '':
            df[config['grouping_name']] = df[config['grouping_name']
                                             ].map(config['class_dic'])
    
        ## Assignment of local variables according to the data necessary
        ## for the neural network.
        train_cols = df.columns [0:-1]
        label = df.columns [-1]
        X = df [train_cols]
        y = df [label]

        ## The prepared data of the dataset is printed.
        print("Dataset used for the explanation.")
        print(df)
        print()
         
        ## The number of elements present in the grouped target is printed.
        print("Number of elements grouped by class.")
        print(y.value_counts())
        print()
         
        ## Application of the oversample to the prepared data.
        print("Oversample application.")
        X, y = data['oversample'].fit_resample(X, y)
        print()
         
        ## The results obtained after applying the oversample are printed.
        print("Number of elements grouped by class.")
        print(y.value_counts())
        print()

        ## Preparation of the parameters of the neural network.
    
        ## We proceed to define the parameters necessary for the training
        ## of the neural network.
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.33, random_state=data['seed'])
    
        ## Neural network training.
    
        ## The alpha parameters and the initial learning rate are adjusted
        ## to obtain the best performance with the model using
        ## cross validation.
    
        ## Local variables are defined for the storage of the scores obtained
        ## by the training.
        
        print("Neural network training.")
        cv_scores_mean = []
        cv_scores_std = []
        regul_param_range = config['regul_param_normal']

        ## Training and validation of different configurations.
        for regul_param in regul_param_range:

            ## Increase the max_iter parameter until it converges.
            mlp = MLPClassifier(hidden_layer_sizes = (10,),
                                activation = 'logistic', solver = 'adam',
                                alpha = regul_param, learning_rate='constant',
                                learning_rate_init = 0.0001, max_iter = 100000,
                                random_state = data['seed'])

            scores = cross_val_score(mlp, X, y, cv = 5, scoring = 'f1_macro')

            cv_scores_mean.append(scores.mean())
            cv_scores_std.append(scores.std())

        ## The results obtained during the training are printed.
        print(cv_scores_mean, cv_scores_std)
        print()

        ## We proceed to draw the learning curve graph according to the data
        ## obtained by the training.
        print("Generation of the learning curve graph.")
        plt.figure()

        ## The average accuracy line is drawn on the test parts.
        plt.plot(np.log10(regul_param_range), cv_scores_mean,
                 color = "g", label = "Test")

        ## The standard deviation band is drawn.
        lower_limit = np.array(cv_scores_mean) - np.array(cv_scores_std)
        upper_limit = np.array(cv_scores_mean) + np.array(cv_scores_std)
        plt.fill_between(np.log10(regul_param_range), lower_limit, upper_limit,
                         color = "#DDDDDD")

        ## Generate the graph.
        plt.title("Learning curve")
        plt.xlabel("Alpha 10^{X}"), plt.ylabel("F1"),
        plt.legend(loc = "best")
        plt.tight_layout()
        plt.savefig("Learning curve Alpha 10^{X} - Iteration " +
                    str(data['iteration']) + ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## The training is performed again keeping the same parameters
        ## except for the alpha value that changes to 1.

        ## Local variables are defined for the storage of the scores
        ## obtained by the training.

        print("Neural network training.")
        cv_scores_mean = []
        cv_scores_std = []
        regul_param_range = config['regul_param_alpha']

        ## Training and validation of different configurations.
        for regul_param in regul_param_range:

            ## Increase the max_iter parameter until it converges.
            mlp = MLPClassifier(hidden_layer_sizes = (10,),
                                activation = 'logistic', solver = 'adam',
                                alpha = 1, learning_rate = 'constant',
                                learning_rate_init = regul_param,
                                max_iter = 100000, random_state = data['seed'])

            scores = cross_val_score(mlp, X, y, cv = 5, scoring = 'f1_macro')

            cv_scores_mean.append(scores.mean())
            cv_scores_std.append(scores.std())

        ## The results obtained during the training are printed.
        print(cv_scores_mean, cv_scores_std)
        print()

        ## We proceed to draw the learning curve graph according to the data
        ## obtained by the training.
        print("Generation of the learning curve graph.")
        plt.figure()

        ## The average accuracy line is drawn on the test parts.
        plt.plot(np.log10(regul_param_range), cv_scores_mean,
                 color = "g", label = "Test")

        ## The standard deviation band is drawn.
        lower_limit = np.array(cv_scores_mean) - np.array(cv_scores_std)
        upper_limit = np.array(cv_scores_mean) + np.array(cv_scores_std)
        plt.fill_between(np.log10(regul_param_range), lower_limit, upper_limit,
                         color = "#DDDDDD")

        ## Generate the graph.
        plt.title("Learning curve")
        plt.xlabel("Learning Rate 10^{-X}"), plt.ylabel("F1"),
        plt.legend(loc = "best")
        plt.tight_layout()
        plt.savefig("Learning curve Rate 10^{-X} - Iteration " +
                    str(data['iteration']) + ".jpg",
                                bbox_inches='tight')
        plt.show()
        print()

        ## Generation of the neural network model.
    
        ## The parameters of the final model are modified according to the
        ## data obtained in the training of the neural network.

        ## The value of the alpha parameter, learning_rate_init and
        ## random_state are changed according to the data in the
        ## config dictionary.
        mlp = MLPClassifier(hidden_layer_sizes = (10,),
                            activation = 'logistic', solver = 'adam',
                            alpha = config['alpha'],
                            learning_rate = 'constant',
                            learning_rate_init = config['learning_rate_init'],
                            max_iter = 100000,
                            random_state = config['random_state'])

        ## We print the data of the final generated model.
        print(mlp.fit(X_train, y_train))
        print()

        ## We safeguard the final generated model
        print("Neural Network Safeguard.")
        joblib.dump(mlp,"model_depression_i" + str(data['iteration']
                                                   ) + ".pkl")
        print("Save model_depression_i" + str(data['iteration']
                                                   ) + ".pkl")
        print()

        ## Generation of the confusion matrix of the model.

        ## We obtain the statistics of the final model to generate its
        ## confusion matrix.

        ## The data for the confusion matrix's generation is defined.
        print("Generation of the confusion matrix.")
        confusion_matrix = ConfusionMatrixDisplay.from_estimator(mlp, X_test,
                           y_test, display_labels = config['target_names'],
                           cmap = plt.cm.Blues)

        confusion_matrix.ax_.set_title("Confusion Matrix")
        plt.savefig("Confusion Matrix - Iteration " +
                    str(data['iteration']) + ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## We get the detailed statistics of the final model.

        ## The necessary detailed prediction is generated.
        y_pred = mlp.predict(X_test)

        ## The detailed statistics of the model are printed.
        print("Obtaining the statistics of the model.")
        print('Classification accuracy =',
              accuracy_score(y_test,y_pred) * 100,'%\n')
        print(classification_report(y_test,y_pred))
        print()

        # The data obtained is saved for later graphing.
        accuracy = accuracy_score(y_test,y_pred) * 100
        clf_rep = precision_recall_fscore_support(y_test,y_pred)

        ## The number of questions used in the iteration is retrieved.
        ds_delete = ds[ds.Status == "Delete"]
        ds_delete.reset_index(inplace=True, drop=True)

        ## All the metrics of the confusion matrix are obtained.
        metrics = [",".join(map(str, ds_delete['Question'].tolist())),
                   accuracy]

        for index in range(0, len(config['name_class'])):
            metrics.append(clf_rep[0][index])
            metrics.append(clf_rep[1][index])
            metrics.append(clf_rep[2][index])
            metrics.append(clf_rep[3][index])

        ## The names of the columns of the dataset are defined.
        columns = ['Question', 'Acurracy global']

        for index in range(0, len(config['name_class'])):
            columns.append('Precision_' + str(index))
            columns.append('Recall_' + str(index))
            columns.append('F1_score_' + str(index))
            columns.append('Support_' + str(index))

        ## A new row of the dataset is generated with all the data.
        data_metrics = pd.DataFrame([metrics], columns = columns)

        ## The data is saved to the dataset.
        print("Safeguarding the model statistics.")
        data_metrics.to_csv(data['path_dataset_qd'], mode = 'a',
                            header = False, index = False)
        print()
        
        ## Model explainability process.

        ## In analysis of the model through the results obtained by ALE,
        ## it is possible to identify the behavior of the neural network.

        ## Slope Analysis of ALE Plots

        ## The necessary parameters for the use of ALE are established
        ## according to the model.
        proba_fun_lr = mlp.predict_proba
        proba_ale_lr = ALE(proba_fun_lr, feature_names = train_cols,
                           target_names = config['target_names'])
        proba_exp_lr = proba_ale_lr.explain(X_train.to_numpy())

        ## We proceed to obtain the graphs to obtain the slopes
        ## of the selected classes.

        ## The question metrics dataset is loaded.
        dm = pd.DataFrame()

        ## The number of items used in the iteration is retrieved.
        ds_item = ds[ds.Status == "In use"]
        ds_item.reset_index(inplace=True, drop=True)

        ## We get the slopes from the ALE data.
        print("Obtaining the slope from the ALE plots.")
        for i in range(df.shape[1]-1):
            structure_data = {'Question': ds_item.loc[i, "Question"]}

            for index in config['selection_class']:
                start_slope = proba_exp_lr.ale_values[i][0][index]
                end_slope = proba_exp_lr.ale_values[i][1][index]
                slope = end_slope - start_slope
                structure_data['Slope ' + 
                               str(config['target_names'][index])] = slope
                structure_data['Threshold ' + 
                               str(config['target_names'][index])] = 'NA'
                structure_data['Anomaly ' + 
                               str(config['target_names'][index])] = 'NA'

            dm = dm.append(structure_data, ignore_index=True)
        print()


        ## Slope analysis and threshold application

        ## The slope obtained is analyzed and those that are above or
        ## below the positive and negative threshold are selected.
        print("The most significant slopes are determined.")
        for i in range(df.shape[1]-1):
            for index in config['selection_class']:
                ## The slopes are selected according to the values of
                ## the thresholds of the defined classes.
                if dm.loc[i, "Slope " + str(config['target_names'][index])
                          ] >= config['positive_threshold']:
                    dm.loc[i, "Threshold " + str(config['target_names'][index])
                           ] = 1
                elif dm.loc[i, "Slope " + str(config['target_names'][index])
                            ] <= config['negative_threshold']:
                    dm.loc[i, "Threshold " + str(config['target_names'][index])
                           ] = 0
        print()

        ## Analysis and determination of anomalous items

        ## The anomalous items present in the model are determined
        ## based on their slope and expected response from the expert.

        ## The dataset of expected responses is loaded.
        da = pd.read_csv(expertKnowledge['path_dataset_qa'])

        ## Delete list of items selected in this iteration.
        i = 0
        for x in ds_delete['Question']:
            da.drop(da.index[x - (i + 1)], axis = 0, inplace = True)
            i += 1

        ## Indexes are restored for later use.
        da.reset_index(inplace=True, drop=True)

        print("Dataset of expected responses.")
        print(da)
        print()

        ## We proceed to identify the anomadic items present in
        ## the model present in this iteration.
        print("A qualification is applied to the data in the dataset.")
        for i in range(df.shape[1]-1):

            ## Local variable to indicate the index of the selected classes.
            i_index = 0

            for index in config['selection_class']:

                ## Determine if it fails based on the expected
                ## response parameters of the defined classes.
                if dm.loc[i, "Threshold " + str(config['target_names'][index])
                          ] == 1 and da.loc[i, "RE"] == 0:
                    dm.loc[i, "Anomaly " + str(config['target_names'][index])
                           ] = config['expected_response_map'][i_index][0]
                if dm.loc[i, "Threshold " + str(config['target_names'][index])
                          ] == 1 and da.loc[i, "RE"] == 1:
                    dm.loc[i, "Anomaly " + str(config['target_names'][index])
                           ] = config['expected_response_map'][i_index][1]
                if dm.loc[i, "Threshold " + str(config['target_names'][index])
                          ] == 0 and da.loc[i, "RE"] == 1:
                    dm.loc[i, "Anomaly " + str(config['target_names'][index])
                           ] = config['expected_response_map'][i_index][2]
                if dm.loc[i, "Threshold " + str(config['target_names'][index])
                          ] == 0 and da.loc[i, "RE"] == 0:
                    dm.loc[i, "Anomaly "  + str(config['target_names'][index])
                           ] = config['expected_response_map'][i_index][3]

                ## Increase of the index of the selected classes.
                i_index += 1
        print()

        print("Iteration rating dataset.")
        print(dm)
        print()

        ## Removing identified anomadic items from the model

        ## We proceed to eliminate the identified anomadic items
        ## so as not to use them in the next iteration.

        ## Local variable for the elimination of the anomaly items.
        instructions = ''
        init = False
        question_anomaly = []

        print("The data to be discarded is determined using the mode: " +
              str(config['selection_mode']))
        if config['selection_mode'] == 1:
            ## Items that fail in the defined class are retrieved.
            dm_delete = dm[dm['Anomaly ' +
                              str(config['target_names'
                              ][config['select_class_mode']])] == 1]
            dm_delete.reset_index(inplace=True, drop=True)
            question_anomaly = dm_delete['Question'].tolist()

        if config['selection_mode'] == 2:
            ## Items that fail in classes defined with the
            ## or condition type are retrieved.
            for index in config['selection_class']:
                if init == False:
                    instructions += "`Anomaly " + str(config[
                        'target_names'][index]) + "` == 1"
                    init = True
                else:
                    instructions += " or `Anomaly " + str(config[
                        'target_names'][index]) + "` == 1"

            dm_delete = dm.query(instructions)
            dm_delete.reset_index(inplace=True, drop=True)
            question_anomaly = dm_delete['Question'].tolist()

        if config['selection_mode'] == 3:
            ## Items that fail in classes defined with the
            ## and condition type are retrieved.
            for index in config['selection_class']:
                if init == False:
                    instructions += "`Anomaly " + str(config[
                        'target_names'][index]) + "` == 1"
                    init = True
                else:
                    instructions += " and `Anomaly " + str(config[
                        'target_names'][index]) + "` == 1"

            dm_delete = dm.query(instructions)
            dm_delete.reset_index(inplace=True, drop=True)
            question_anomaly = dm_delete['Question'].tolist()

        if config['selection_mode'] == 4:
            ## Out-of-threshold items are retrieved in
            ## classes defined with the condition type and.
            for index in config['selection_class']:
                if init == False:
                    instructions += "`Anomaly " + str(config[
                        'target_names'][index]) + "` == 'NA'"
                    init = True
                else:
                    instructions += " and `Anomaly " + str(config[
                        'target_names'][index]) + "` == 'NA'"

            dm_delete = dm.query(instructions)
            dm_delete.reset_index(inplace=True, drop=True)
            question_anomaly = dm_delete['Question'].tolist()
        print()

        ## The selected blank questions are printed according to the
        ## config['selection_mode'].
        print("List of discarded data.")
        print(question_anomaly)
        print()
        
        # The data used for the test of the neural network model is shown.
        print("Model test data.")
        print(X_test)
        print()

        ## Explanation of the model using ALE.

        ## In analysis of the model through the results obtained by ALE,
        ## it is possible to identify the behavior of the neural network.

        ## The necessary parameters for the use of ALE are established
        ## according to the model.

        proba_fun_lr = mlp.predict_proba
        proba_ale_lr = ALE(proba_fun_lr, feature_names = train_cols,
                           target_names = config['target_names'])
        proba_exp_lr = proba_ale_lr.explain(X_train.to_numpy())

        # The graphs of all the data present in the dataset used are shown.
        print("Generation of ALE graphs.")
        fig, ax = plt.subplots()
        plot_ale(proba_exp_lr, n_cols=2, features=list(range(df.shape[1]-1)),
                 ax=ax, fig_kw={'figwidth': 10, 'figheight': 180})
        plt.savefig("ALE graphs - Iteration " + str(data['iteration']) +
                    ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Explanation of the model using LIME.

        ## We proceed to explain the improved model using LIME for all classes.

        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
                                            categorical_features = train_cols,
                                            feature_names=train_cols,
                                            class_names=config['target_names'],
                                            discretize_continuous=True)

        ## The first, second and last individual present in the test
        ## data is explained.
        single_relationship = ['first', 'second', 'latest']
        map_relationship = [0,1,-1]
        for individual_index in map_relationship:
            print("Generation of the LIME explanation graph of the " +
                  single_relationship[map_relationship.index(
                      individual_index)] +
                  " individual")
            exp = explainer.explain_instance(X_test.to_numpy()[
                individual_index], mlp.predict_proba, num_features=6,
                top_labels=1)

            for i in range(0, len(config['target_names'])):
                try:
                    exp.as_pyplot_figure(label=i)
                    plt.savefig("LIME graphs of the " + 
                                single_relationship[map_relationship.index(
                                individual_index)] + " - Iteration " +
                                str(data['iteration']) + ".jpg",
                                bbox_inches='tight')
                    plt.show()
                    break
                except:
                    pass
            print()

        ## Explanation of the model using SHAP.

        ## SHAP explanatory values are displayed according to the
        ## class to which the selected individual belongs according to
        ## the prediction of the model.

        ## The first, second and last individual present in the test
        ## data is explained.
        single_relationship = ['first', 'second', 'latest']
        map_relationship = [0,1,-1]
        for individual_index in map_relationship:
            print("Generation of the SHAP explanation graph of the " +
                  single_relationship[map_relationship.index(
                      individual_index)] +
                  " individual")
            individual = X_test.iloc[[individual_index]]

            ## Individual explanation by SHAP.
            explainer = shap.KernelExplainer(mlp.predict_proba, X_train,
                                    feature_names = train_cols,
                                    output_names = config['target_names'])
            shap_values = explainer.shap_values(individual)
            clase = mlp.predict(individual)[0]

            fig = plt.gcf()
            shap.summary_plot(shap_values[clase], individual,
                              feature_names = train_cols, plot_type = "bar")
            fig.savefig("SHAP graphs of the " + 
                        single_relationship[map_relationship.index(
                        individual_index)] + " - Iteration " +
                        str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            print()

        ## Explanation using heatmaps.

        ## The importance of the data present in the neural network is shown
        ## from explanation methods.

        ## Questions currently in use are retrieved.
        ds_item = ds[ds.Status == "In use"].reset_index(drop=True)

        ## The data is prepared for the generation of heatmaps.
        column_Map = ds_item['Question'].tolist()
        content_Map = [False for x in range(1, df.shape[1])]

        ## The next items to be deleted will be marked.
        for index in range(0, len(content_Map)):
            if data['iteration'] == 0:
                for delete in config['list_contr']:
                    if column_Map[index] == delete:
                        content_Map [index] = True
            else:
                for delete in question_anomaly:
                    if column_Map[index] == delete:
                        content_Map [index] = True

        ## The dataframe is created with the prepared data.
        df_mask = pd.DataFrame([content_Map], columns = column_Map)
        df_mask_imp = pd.concat([df_mask, df_mask, df_mask],
                                ignore_index = True, axis = 0)
        df_mask_all = pd.concat([df_mask, df_mask, df_mask, df_mask],
                                ignore_index = True, axis = 0)

        ## The heatmap mask dataset is printed.
        print("heatmap mask dataset.")
        print(df_mask)
        print()

        ## Identification of feature variables.

        ## Identification of feature variables from the randomness of the
        ## answers as questions of the data.

        ## Loading the data for explainability.
        explainer = ClassifierExplainer(mlp, X_test, y_test)

        ## Class index local variable.
        map_class = []

        for iterable_class in range(0, len(config['name_class'])):
            ## Explainability of the data by means of the heat-
            ## map of all the classes.
            print(config['name_class'][iterable_class] +
                  " Important Variable heatmap generation.")
            df_importance = explainer.permutation_importances(iterable_class
                                                              ).sort_index()
            df_importance.index = column_Map
            df_importance.drop("Score",inplace=True, axis=1)
            df_importance.drop("Feature",inplace=True, axis=1)
            df_importance.loc[df_importance['Importance'] < 0,
                              'Importance'] = 0
            map_class.append(df_importance)

            ## Generation of the heatmap.
            plt.figure(figsize = (28,6))
            sns.heatmap(map_class[iterable_class].transpose(),
                        cmap="Reds", cbar = False)
            sns.heatmap(map_class[iterable_class].transpose(),
                        cmap="Blues", yticklabels = True, xticklabels = True,
                        mask = df_mask.to_numpy())
            plt.savefig(config['name_class'][iterable_class] +
                  " Important Variable heatmap - Iteration " +
                  str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

        ## Merging of all the classes heatmaps into one map.

        ## Fusion of the previous heatmaps.
        df_importance = pd.concat(map_class, axis=1)
        df_importance.columns = config['name_class']

        ## Generation of the heatmap.
        print("Generation of the heatmap of Important Variable of" +
              " all classes.")
        plt.figure(figsize = (32,6))
        sns.heatmap(df_importance.transpose(), cmap = "Reds", cbar = False)
        sns.heatmap(df_importance.transpose(), cmap = "Blues",
                    yticklabels = True, xticklabels = True ,
                    mask = df_mask_imp.to_numpy())
        plt.savefig("All classes of Important Variable heatmap - Iteration " +
        str(data['iteration']) + ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Identification of important variables using SHAP.

        # Contribution in the final probability of each question for specific
        ## instances at the local level. For each class, a map is first shown
        ## with all the individuals and the contribution of each question,
        ## and then a map made with the mean of these values in absolute value.
        ## This seeks to globalize the scope of SHAP.

        ## Class index local variable.
        map_class = []

        ## Explainability of the data by means of the heatmap
        ## of all the classes.

        ## Definition of explainability data.

        for iterable_class in range(0, len(config['name_class'])):
            df_shap = explainer.get_shap_values_df(iterable_class)
            df_shap.columns = column_Map

            ## Generation of the heatmap.
            print(config['name_class'][iterable_class] +
                  " SHAP heatmap generation.")
            plt.figure(figsize = (28,14))
            sns.heatmap(df_shap, cmap="vlag_r", yticklabels=True,
                        xticklabels=True, center = 0)
            plt.savefig(config['name_class'][iterable_class] +
                  " SHAP heatmap - Iteration " + 
                  str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

            ## Fusion of the data used.

            df_shap_mean = df_shap.abs().mean(axis=0).to_frame()
            df_shap_mean.columns= ["Mean SHAP"]
            map_class.append(df_shap_mean)

            ## Generation of the heatmap.
            print(config['name_class'][iterable_class] +
                  " SHAP average heatmap generation.")
            plt.figure(figsize = (28,6))
            sns.heatmap(map_class[iterable_class].transpose(), cmap="Reds",
                        cbar = False)
            sns.heatmap(map_class[iterable_class].transpose(), cmap="Blues",
                        yticklabels = True, xticklabels = True,
                        mask = df_mask.to_numpy())
            plt.savefig(config['name_class'][iterable_class] +
                  " SHAP average heatmap - Iteration " +
                  str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

        ## Merging of all the classes heatmaps into one map.

        ## Fusion of the previous heatmaps.
        df_shap = pd.concat(map_class, axis=1)
        df_shap.columns = config['name_class']

        ## Generation of the heatmap.
        print("Generation of the heatmap of SHAP of all classes.")
        plt.figure(figsize = (32,6))
        sns.heatmap(df_shap.transpose(), cmap="Reds", cbar = False)
        sns.heatmap(df_shap.transpose(), cmap="Blues", yticklabels = True,
                    xticklabels = True, mask = df_mask_imp.to_numpy())
        plt.savefig("All classes of SHAP heatmap - Iteration " + 
                    str(data['iteration']) + ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Identification of feature variables using LIME.

        ## Contribution in the final probability of each question for
        ## specific instances at the local level. For each class, a map
        ## is first shown with all the individuals and the contribution
        ## of each question, and then a map made with the mean of these values
        ## in absolute value. This seeks to globalize the scope of LIME.

        ## Data preparation for explainability.
        lime_exp = lime.lime_tabular.LimeTabularExplainer(X_train.to_numpy(),
                categorical_features = train_cols, feature_names=train_cols,
                class_names=config['target_names'], discretize_continuous=True)

        ## Class index local variable.
        map_class = []

        ## Obtaining the data for the generation of the explanations by LIME
        ## about all the classes.

        ## Local variables for LIME array data.
        exp_matrix = [[] for _ in range(len(config['name_class']))]

        ## Recovering the data needed for LIME.
        for x in X_test.to_numpy():

            ## Auxiliary temporary variables.
            exp_list = [[] for _ in range(len(config['name_class']))]

            ## Loading the data for explainability.
            exp = lime_exp.explain_instance(x, mlp.predict_proba,
                                    num_features = df.shape[1] - 1,
                                    top_labels = len(config['name_class']))

            ## Recovering the data.
            for elements in range(0, len(config['name_class'])):
                temp = exp.as_map()[elements]
                temp.sort(key=itemgetter(0))

                ## Saving data of all the classes.
                for tup in temp:
                    exp_list[elements].append(tup[1])

                exp_matrix[elements].append(exp_list[elements])

        ## Explainability of the data by means of the heatmap of
        ## all the classes.

        # Generation of the heatmap.

        for iterable_class in range(0, len(config['name_class'])):
            print(config['name_class'][iterable_class] +
                  " LIME heatmap generation.")
            plt.figure(figsize = (28,14))
            lime_df = pd.DataFrame(exp_matrix[iterable_class])
            lime_df.columns = column_Map
            sns.heatmap(lime_df, cmap="vlag_r",yticklabels=True,
                        xticklabels=True)
            plt.savefig(config['name_class'][iterable_class] +
                  " LIME heatmap - Iteration " + 
                  str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

            ## Fusion of the data used.

            lime_df_mean = lime_df.abs().mean(axis = 0).to_frame()
            lime_df_mean.columns= ["Mean LIME"]
            map_class.append(lime_df_mean)

            print(config['name_class'][iterable_class] +
                  " LIME average heatmap generation.")
            plt.figure(figsize = (32,6))
            sns.heatmap(map_class[iterable_class].transpose(), cmap="Reds",
                        cbar = False)
            sns.heatmap(map_class[iterable_class].transpose(), cmap="Blues",
                        yticklabels = True, xticklabels = True,
                        mask = df_mask.to_numpy())
            plt.savefig(config['name_class'][iterable_class] +
                  " LIME average heatmap - Iteration " + 
                  str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

        ## Merging of all the classes heatmaps into one map.
        print("Generation of the heatmap of LIME of all classes.")
        df_lime = pd.concat(map_class, axis=1)
        df_lime.columns = config['name_class']
        plt.figure(figsize = (32,6))
        sns.heatmap(df_lime.transpose(), cmap="Reds", cbar = False)
        sns.heatmap(df_lime.transpose(), cmap="Blues", yticklabels = True,
                    xticklabels = True, mask = df_mask_imp.to_numpy())
        plt.savefig("All classes of LIME heatmap - Iteration " + 
                    str(data['iteration']) + ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Identification of feature variables using ALE.

        ## It shows the main effect of each question compared to the
        ## probability that the model predicts in each class. This effect is
        ## shown for both responses. Since there are only 2 answers, the map
        ## is the same but with the colors reversed.

        ## Definition of explainability data.
        proba_fun_lr = mlp.predict_proba
        proba_ale_lr = ALE(proba_fun_lr, feature_names=train_cols,
                           target_names = config['target_names'])
        proba_exp_lr = proba_ale_lr.explain(X_train.to_numpy())

        ## Class index local variable.
        map_class = []

        ## Obtaining the data for the generation of the explanations
        ## by ALE about all the classes.

        ## Local variables for ALE array data.
        ale_list = [[] for _ in range(len(config['name_class']))]

        ## Recovering the data needed for ALE.
        for array in proba_exp_lr.ale_values:

            ## Recovering the data.
            for elements in range(0, len(config['name_class'])):

                ## Saving data of all the classes.
                ale_list[elements].append(array[0][elements])

        ## Data processing for the explainability of ALE.

        ## Data processing of all the classes.
        for elements in range(0, len(config['name_class'])):
            ale_df = pd.DataFrame([ale_list [elements]])
            ale_df = pd.concat([ale_df.multiply(-1), ale_df])
            ale_df.index = [config['name_class'] [elements] + " - False",
                            config['name_class'] [elements] + " - True"]
            ale_df.columns = column_Map
            map_class.append(ale_df)

        ## Explainability of the data by means of the heatmap
        ## of all the classes.

        ## Generation of the heatmap.

        for iterable_class in range(0, len(config['name_class'])):
            print(config['name_class'][iterable_class] +
                  " ALE heatmap generation.")
            plt.figure(figsize = (28,6))
            sns.heatmap(map_class[iterable_class], cmap="vlag_r",
                        yticklabels=True, xticklabels=True)
            plt.savefig(config['name_class'][iterable_class] +
                  " ALE heatmap - Iteration " + 
                  str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

        ## Fusion of the data used.

        ## Application of absolute value to data.
        for index in range(0, len(map_class)):
            map_class [index] = map_class [index][0:1].abs()

        ## Union of the heatmaps of all the class.
        ale_df = pd.concat(map_class)
        ale_df.index = config['name_class']
        ale_df = ale_df.transpose()

        # Generation of the heatmap.

        print("Generation of the heatmap of ALE of all classes.")
        plt.figure(figsize = (28,6))
        sns.heatmap(ale_df.transpose(), cmap="Reds", cbar = False)
        sns.heatmap(ale_df.transpose(), cmap="Blues", yticklabels = True,
                    xticklabels = True, mask= df_mask_imp.to_numpy())
        plt.savefig("All classes of ALE heatmap - Iteration " + 
                    str(data['iteration']) + ".jpg",
                    bbox_inches='tight')
        plt.show()
        print()

        ## Comparing of Feature Variables Methods for Explainability

        ## The global values of each of the methods used are taken,
        ## their maximums are obtained (the minimums are 0) and a
        ## percentage is calculated based on said maximum, which
        ## indicates how relevant the contribution is for each question.

        # ### All classes method comparison

        # The highest data of each method used is obtained.

        for iterable_class in range(0, len(config['name_class'])):
            ## The maximum present value is obtained in each explainability
            ## method.
            max_importance = df_importance[config['name_class'][iterable_class]
                                           ].max()
            max_shap = df_shap[config['name_class'][iterable_class]].max()
            max_lime = df_lime[config['name_class'][iterable_class]].max()
            max_ale  = ale_df[config['name_class'][iterable_class]].max()

            # A new heatmap is created containing all the values of each
            ## method used.

            ## The values are calculated to plot the new heatmap.
            data_importance = df_importance[config[
                'name_class'][iterable_class]].multiply(100/max_importance)
            data_lime = df_lime[config['name_class'][iterable_class]
                                ].multiply(100/max_lime)
            data_shap = df_shap[config['name_class'][iterable_class]
                                ].multiply(100/max_shap)
            data_ale = ale_df[config['name_class'][iterable_class]
                             ].multiply(100/max_ale)

            df_general = pd.DataFrame([data_importance, data_lime, data_shap,
                                       data_ale])

            ## Method names are assigned.
            df_general.index = data['name_methods']
            df_general

            # Generation of the heatmap.
            print("Generation of the heatmap of all the comparison methods" +
                  " of the " + config['name_class'][iterable_class] + ".")
            plt.figure(figsize = (28,8))
            sns.heatmap(df_general, cmap="Reds", cbar = False)
            sns.heatmap(df_general, cmap="Blues", yticklabels = True,
                        xticklabels = True, mask = df_mask_all.to_numpy())
            plt.savefig("Comparison of all the methods of the " + 
                        config['name_class'][iterable_class] + 
                        " - Iteration " + 
                        str(data['iteration']) + ".jpg",
                        bbox_inches='tight')
            plt.show()
            print()

        ## Safeguarding the items eliminated during this iteration.

        ## Removal of anomalous items selected by selection_mode.
        for x in question_anomaly:

            ## Status change of the items question in the respective
            ## auxiliary dataset.
            ds.loc[x - 1,'Status'] = 'Delete'

        ## Save the changed statuses in the dataset.
        print("Safeguarded the data determined from the status dataset.")
        ds.to_csv(data['path_dataset_qs'], index = False)
        print()

        print("Ending iteration number " + str(data['iteration']) + ".")
        print()

        ## If the determined data list is zero.
        if len(question_anomaly) == 0:
            ## Then the iterative improvement process ends.
            print("End of iteration process.")
            print()
            break

    ## Graphing of the statistics obtained.

    ## Loading data from the dataset.
    df = pd.read_csv(model['path_dataset'])
    dd = pd.read_csv(data['path_dataset_qd'])

    ## Elimination of unrelated items from the dataset.
    if len(config['unrelated_questions']) != 0:
        df = df.drop(df.columns[config['unrelated_questions'][0]:
                                config['unrelated_questions'][1]],
                     axis=1)

    ## All the data necessary to generate the graphs is obtained.

    ## Local variables for graph generation.
    total_question = []
    global_accuracy = []
    total_delete = []
    recalls = [[] for _ in range(len(config['name_class']))]
    labels_plot = []
    
    ## Get the total number of iterationes.
    iterations = list(map(str, list(range(1, (dd.shape[0] + 1)))))
    
    ## Obtaining data from the dataset.
    for x in range(dd.shape[0]):
        
        ## The total number of items used in each iteration is obtained.
        total_question.append((df.shape[1] - 1) - len((dd.loc[x, 'Question'
                                                             ]).split(',')))
        
        ## The global acurracy of each data['iteration'] is obtained.
        global_accuracy.append((dd.loc[x, 'Acurracy global']) / 100)
        
        ## The total number of items eliminated in each iteration
        ## is obtained.
        if x == 0:
            total_delete.append(len((dd.loc[x, 'Question']).split(',')))
        else:
            data_delete =len((dd.loc[x, 'Question']).split(',')) - len((dd.loc[
                x - 1, 'Question']).split(','))
            total_delete.append(data_delete)
        
        ## The recall of all classes of each iteration
        ## is obtained.
        for index in range(0, len(config['name_class'])):
            recalls[index].append(dd.loc[x, 'Recall_' + str(index)])
        
    for target in config['target_names']:
        labels_plot.append(target + " Class")

    ## Generation of the graph of items used in each iteration.

    ## Generation of the graph according to the data obtained.
    print("Graph of questions used in each iteration.")
    plt.figure()
    labels = ['Remaining questions']
    plt.title('Remaining questions over several item removal iterations')
    plt.plot(iterations, total_question, marker= 'o', color = 'blue')
    plt.ylabel("Remaining questions")
    plt.xlabel("Iterations")
    plt.ylim(0, df.shape[1] - 1)
    plt.legend(labels)
    plt.savefig("Questions used in each iteration.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Generation of the graph of the acurracy obtained in each iteration.

    ## Generation of the graph according to the data obtained.
    print("Graph of the acurracy obtained in each iteration.")
    plt.figure()
    labels=['Accuracy']
    plt.title('Accuracy results obtained over several item removal iterations')
    plt.plot(iterations, global_accuracy, marker= 'o', color = 'red')
    plt.ylabel("Accuracy")
    plt.xlabel("Iterations")
    plt.ylim(0, 1)
    plt.legend(labels)
    plt.savefig("Questions used in each iteration.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## Generation of the graph of items eliminated in each iteration.

    ## Generation of the graph according to the data obtained.
    print("Graph of questions eliminated in each iteration.")
    plt.figure()
    labels=['Anomalous items']
    plt.title('Amount of anomalous items obtained over iterations')
    plt.plot(iterations, total_delete, marker= 'o', color = 'green')
    plt.xlabel("Iterations")
    plt.ylabel("Anomalous items")
    plt.legend(labels)
    plt.savefig("Questions eliminated in each iteration.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## The graph is generated with the merger of the acurracy with the
    ## anomadic items eliminated.

    ## Generation of the graph according to the data obtained.
    print("Accuracy plot with nomadic and deleted questions.")
    fig, ax = plt.subplots()

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Accuracy', color = 'tab:red')
    ax.plot(iterations, global_accuracy, marker= 'o', color= 'tab:red')
    ax.tick_params(axis='y', labelcolor= 'tab:red')

    plt.ylim(0, 1)
    ax = ax.twinx()  

    ax.set_ylabel('Anomalous Items', color= 'tab:green')  
    ax.plot(iterations, total_delete, marker= '^', color= 'tab:green')
    ax.tick_params(axis='y', labelcolor= 'tab:green')

    fig.tight_layout()
    plt.title('Accuracy and number of anomalous questions' +
              'obtained over several item removal iterations')
    plt.xticks(range(0, data['iteration'], 1))
    plt.savefig("Accuracy plot with nomadic and deleted questions.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## The graph is generated with the fusion of the recall obtained in
    ## the improvement process.

    ## Generation of the graph according to the data obtained.
    print("Graph of the recall obtained in the improvement process.")
    plt.figure()
    plt.title('Recall results obtained over several item removal iterations')
    for index in range(0, len(config['name_class'])):
        plt.plot(iterations, recalls[index], marker= 'o',
                 color = config['list_color'][index])
    plt.xlabel("Iterations")
    plt.ylabel("Recall")
    plt.ylim(0, 1)
    plt.legend(labels_plot)
    plt.savefig("Recall obtained in the improvement process.jpg",
                bbox_inches='tight')
    plt.show()
    print()

    ## The graph is generated with the fusion of the recall and the eliminated
    ## questions obtained in the improvement process.

    ## Generation of the graph according to the data obtained.
    print("Graph of the fusion of the recall and the questions eliminated.")
    fig, ax = plt.subplots()

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Recall', color= 'tab:blue')
    for index in range(0, len(config['name_class'])):
        ax.plot(iterations, recalls[index], marker= 'o',
                 color = config['list_color'][index])
    ax.tick_params(axis='y', labelcolor= 'tab:blue')

    plt.ylim(0, 1)
    plt.legend(labels_plot)
    ax = ax.twinx()

    ax.set_ylabel('Anomalous Items', color = 'tab:green')  
    ax.plot(iterations, total_delete, marker= '^', color = 'tab:green')
    ax.tick_params(axis='y', labelcolor = 'tab:green')

    fig.tight_layout()
    plt.title('Recall results and anomalous questions obtained ' +
              'over several item removal iterations')
    plt.xticks(range(0, data['iteration'], 1))
    plt.savefig("Fusion of the recall and the questions eliminated.jpg",
                bbox_inches='tight')
    plt.show()
    print()

## Main of the script.
loadImports()
data = loadData()
model = loadModel()
config = configParameters()
expertKnowledge = loadExpectedOutputs()
explain(data, model, config, expertKnowledge)
