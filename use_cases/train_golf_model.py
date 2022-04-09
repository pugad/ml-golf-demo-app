import pandas as pd
from models.decision_tree_classifier import DecisionTreeClassifierModel
from use_cases.initialize_model import initialize_model

def train_dt_classifier(ml_model:DecisionTreeClassifierModel, features_df:pd.DataFrame, labels:list):
    '''
    Trains a given decision tree classifier model using the provided features dataset and labels,
    then returns the trained classifier.
    params:
        -   ml_model:       DecisionTreeClassifierModel
        -   features_df:    pandas DataFrame containing the training data (also known as features). Categorical values are already one-hot encoded.
        -   labels:        list of expected training set results (also known as labels)
    '''
    return ml_model.train(X_train=features_df, y_train=labels)



def initialize_train_persist(model_id:str, features_df:pd.DataFrame, labels:list, persist_dir:str):
    '''
    Initializes the model, trains it, then persists it.
    params:
        -   model_id:       string representing the instantiated model's ID
        -   features_df:    pandas DataFrame containing the training data (also known as features). Categorical values are already one-hot encoded.
        -   labels:         list of expected training set results (also known as labels)
        -   persist_dir:    string path to the directory to persist the model on disk
    '''

    # initialize the model. The initialized models will automatically
    # be persisted in the running container.
    golf_dt_clf = initialize_model(
        model_id=model_id,
        modelClass=DecisionTreeClassifierModel,
        persist_dir=persist_dir
    )

    # train the model
    train_dt_classifier(golf_dt_clf, features_df=features_df, labels=labels)

    # persist the trained model
    golf_dt_clf.persist(persist_dir=persist_dir)

    return golf_dt_clf