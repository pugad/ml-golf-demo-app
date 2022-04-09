from models.basemodel import BaseMLModel

def initialize_model(model_id:str, modelClass:BaseMLModel, persist_dir:str):
    '''
    Initializes an ML model by class, then trains it.
    params:
        -   model_id:           an ID to uniquely identify an instance of this model
        -   modelClass:         ML model class (based on BaseMLModel) to instantiate
        -   persist_dir:        directory to persist the model in
    
    returns:
        -   model:              initialized model
    '''
    model = modelClass(model_id=model_id, persist_dir=persist_dir)
    return model