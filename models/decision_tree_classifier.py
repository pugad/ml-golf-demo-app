from typing import Any
from sklearn import tree
import pandas
from .basemodel import BaseMLModel
import joblib
import uuid
import os

class DecisionTreeClassifierModel(BaseMLModel):
    '''
    An abstraction to hold tree.DecisionTreeClassifier
    params:
        -   model_id:           ID of the model once instantiated
        -   clf:                (optional) load an already-instantiated model
        -   persist_dir:        directory to persist the model in
    '''
    def __init__(self, model_id:str=str(uuid.uuid4())[:8], clf:tree.DecisionTreeClassifier=None, persist_dir=None):
        # set the id
        self.model_id = model_id
        
        # initialize the classifier
        self.clf = clf if clf is not None else tree.DecisionTreeClassifier()

        # persist
        if persist_dir:
            self.persist(persist_dir=persist_dir)
    
    def train(self, X_train:pandas.DataFrame, y_train:list) -> tree.DecisionTreeClassifier:
        '''
        Train the Decision Tree Classifier
        params:
            -   X_train:    features as a pandas DataFrame (or numpy array) containing numerical data (categorical data are one-hot encoded)
            -   y_train:    labels/targets as a list containing the expected results (to train our model with).
        returns:
            -   trained tree.DecisionTreeClassifier
        '''
        # train model
        clf_trained = self.clf.fit(X=X_train, y=y_train)
        
        # update model
        self.clf = clf_trained
        
        # return the trained classifier
        return self.clf
    
    def predict(self, X_test:pandas.DataFrame) -> Any:
        '''
        Predict using the trained classifier
        params:
            -   X_test: features pandas DataFrame (or numpy array) containing data to predict with
        returns:
            -   predicted value with the same data type as the labels data used for training
        '''
        
        return self.clf.predict(X_test)
    
    def persist(self, persist_dir):
        '''
        Persist this model in disk using joblib.dump

        Note: demo purposes only; may not be good for production
        '''
        joblib.dump(self.clf, os.path.join(persist_dir, self.model_id + '.joblib'))
    
    @classmethod
    def load_model(self, persist_dir, model_id):
        '''
        Load a persisted model from disk using joblib.load

        Note: demo purposes only; may not be good for production
        '''
        model_filepath = os.path.join(persist_dir,model_id+'.joblib')

        return self(model_id=model_id, clf=joblib.load(model_filepath))