from pydantic import BaseModel
import typing
import pandas as pd

class TrainDataModelObservation(BaseModel):
    '''
    Data model to validate each observation in a POST payload to /train
    '''
    outlook: str
    temperature: int
    humidity: float
    windy: bool
    play: str

    def validate_and_get(self):
        '''
        Validates whether the categorical variables
        are within the expected set of choices.
        '''
        if not self.outlook in ['sunny','rainy','overcast']:
            return {}

        self.windy = str(self.windy).lower()
        return self.dict()

class TrainDataModel(BaseModel):
    '''
    Data model to validate the actual payload sent to /train
    '''
    data: typing.List[TrainDataModelObservation]

    def to_dataframe(self):
        return pd.DataFrame.from_records([obs.validate_and_get() for obs in self.data]).dropna().reset_index(drop=True)
    
        

class PredictDataModel(BaseModel):
    '''
    Data model to validate a POST payload to /predict
    '''
    outlook: str
    temperature: int
    humidity: float
    windy: bool

    def validate_and_get(self):
        '''
        Validates whether the categorical variables
        are within the expected set of choices.
        '''
        if not self.outlook in ['sunny','rainy','overcast']:
            return {}

        self.windy = str(self.windy).lower()
        return self.dict()
    
    def to_dataframe(self):
        return pd.DataFrame.from_records([self.validate_and_get()]).dropna()
    
