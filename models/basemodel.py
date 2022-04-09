

class BaseMLModel:
    '''
    Base ML Model class
    params:
        -   model_id:           ID of the model once instantiated
        -   persist_dir:        filepath to persist the model in
    '''
    def __init__(self, model_id:str, persist_dir:str=None, *args, **kwargs):
        self.model_id = model_id
        if persist_dir:
            self.persist(persist_dir=persist_dir)

    def train(self, X_train, y_train, *args, **kwargs):
        pass

    def predict(self, X_test, *args, **kwargs):
        pass

    def persist(self, persist_dir, *args, **kwargs):
        pass
    
    def load_model(self, model_filepath, *args, **kwargs):
        pass