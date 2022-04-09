import pandas
from models.decision_tree_classifier import DecisionTreeClassifierModel

def predict_play_golf(trained_ml_model:DecisionTreeClassifierModel, inputs:pandas.DataFrame) -> str:
    '''
    Returns a prediction based on the given inputs and the trained classifier
    params:
        -   trained_ml_model: a trained DecisionTreeClassifierModel instance
        -   inputs: pandas DataFrame containing inputs with the same variable names as the trained model's
    
    returns:
        -   string of the predicted value
    '''
    return str(trained_ml_model.predict(X_test=inputs))