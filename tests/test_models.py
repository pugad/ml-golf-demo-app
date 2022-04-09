from models.basemodel import BaseMLModel
from models.decision_tree_classifier import DecisionTreeClassifierModel


def test_base_model():
    assert 'train' in dir(BaseMLModel)
    assert 'predict' in dir(BaseMLModel)
    assert 'persist' in dir(BaseMLModel)
    assert 'load_model' in dir(BaseMLModel)

def test_decision_tree_classifier():
    assert 'train' in dir(DecisionTreeClassifierModel)
    assert 'predict' in dir(DecisionTreeClassifierModel)
    assert 'persist' in dir(DecisionTreeClassifierModel)
    assert 'load_model' in dir(DecisionTreeClassifierModel)

    # TODO: add more specific tests
