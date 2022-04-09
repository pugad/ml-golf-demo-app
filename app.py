from models.decision_tree_classifier import DecisionTreeClassifierModel
from models.datamodels import TrainDataModel, PredictDataModel

from use_cases.train_golf_model import initialize_train_persist

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datasets.wrapper import get_weather_data

import pandas as pd
import os
import uuid

# Initialize FastAPI.
APP = FastAPI()

# Mount static and templates folders.
APP.mount('/static', StaticFiles(directory='static'), name='static')
TEMPLATES = Jinja2Templates(directory='templates')

# Specify the path to the initial training dataset.
WEATHER_DATASET = 'datasets/weather_data.csv'

# For the sake of this demo, let's just make the models persist in a folder here.
# In production, separate databases or apps could manage the storage of persisted models
# to make it highly available.
PERSISTENCE_DIR = 'persist_dir'
if not os.path.exists(PERSISTENCE_DIR):
    os.mkdir(PERSISTENCE_DIR)

# Initialize model ID/s.
GOLF_MODEL_1 = 'golf-mlmodel-' + str(uuid.uuid4())[:8]

# Initialize, train, and persist the model.
features_df, labels = get_weather_data(WEATHER_DATASET)
initialize_train_persist(
    model_id=GOLF_MODEL_1,
    features_df=features_df,
    labels=labels,
    persist_dir=PERSISTENCE_DIR
)

# TODO: perform initial model evaluation here
# <-code to evaluate trained model/s->


@APP.get("/")
async def home(request:Request):
    return TEMPLATES.TemplateResponse("index.html", context={'request':request}, status_code=200)

@APP.get("/healthz")
async def healthz():
    '''
    Healthz endpoint to tell a K8s cluster that our app isn't ready to be served.
    '''
    # For this demo, we'll just check if the weather dataset exists.
    # In production, we can ping the persistent storage provider here.
    # If it fails, we return a non-2XX status code
    # (maybe a 503 Service Unavailable)
    if not os.path.exists(WEATHER_DATASET):
        return JSONResponse(content={
            'status':'not ready',
            'reason':'persistent storage unavailable'
        }, status_code=503)
    
    return JSONResponse(content={'status':'ok'}, status_code=200)

@APP.post("/train")
async def train(data:TrainDataModel):
    '''
    Trains the model using data received externally
    '''

    # Validate obtained training data.
    new_train_df = data.to_dataframe()
    if not len(new_train_df):
        return JSONResponse(
            content={'status':'error','reason':'invalid/empty data'},
            status_code=400
        )
    

    # The implementation below isn't scalable
    # but it will suffice in this example
    # since we know we are using a small dataset.
    # We'll likely have to switch to a random forest
    # classifier and use dask for larger datasets.

    # Get features and perform any necessary one-hot encoding and cleaning.
    new_features_df = new_train_df[['temperature','humidity']]
    new_labels = new_train_df['play'].tolist()
    new_one_hot_encoded_df = pd.DataFrame({
        'outlook_sunny':[],
        'outlook_rainy':[],
        'outlook_overcast':[],
        'windy_false':[],
        'windy_true':[]
    }).append(pd.get_dummies(new_train_df[['outlook','windy']])).fillna(0).astype(int)
    new_features_df = new_features_df.join(new_one_hot_encoded_df)

    # Reload any persisted models here.
    # This is actually not necessary for a DecisionTreeClassifier
    # since we'll have to retrain it with the full dataset again anyway.
    # But for other models that can do partial fitting, persisting and reloading it
    # will save us time.
    golf_dt_clf = DecisionTreeClassifierModel.load_model(persist_dir=PERSISTENCE_DIR, model_id=GOLF_MODEL_1)

    # We reload the initial/current weather dataset and combine it with the new training dataset
    # since we're using a DecisionTreeClassifier.
    current_features_df, current_labels = get_weather_data(WEATHER_DATASET)
    prev_feature_count = len(current_features_df)
    prev_labels_count = len(current_labels)
    current_features_df = current_features_df.append(new_features_df).reset_index(drop=True)
    current_labels.extend(new_labels)

    # Retrain the model.
    golf_dt_clf.train(X_train=current_features_df, y_train=current_labels)
    
    # TODO: Perform model evaluation and testing here.
    # <-code to evaluate and test trained model->

    # If trained model fails expectations,
    # it will not be persisted.
    # Instead we could log the results.
    # e.g. submit evaluation results/metrics to
    # an external logging service/database
    
    # Otherwise, persist the trained model.
    golf_dt_clf.persist(PERSISTENCE_DIR)

    return JSONResponse(content={
        'status':'model trained',
        'previous_features_count':prev_feature_count,
        'previous_labels_count':prev_labels_count,
        'new_features_count':len(current_features_df),
        'new_labels_count':len(current_labels)
    }, status_code=200)

@APP.post("/predict")
async def predict(data:PredictDataModel):
    '''
    Performs a prediction using the trained and persisted model.
    '''

    # Validate input data.
    predict_df = data.to_dataframe()
    if not len(predict_df):
        return JSONResponse(
            content={'status':'error','reason':'invalid/empty data'},
            status_code=400
        )

    # Perform one-hot encoding.
    predict_one_hot_encoded_df = pd.DataFrame({
        'outlook_sunny':[],
        'outlook_rainy':[],
        'outlook_overcast':[],
        'windy_false':[],
        'windy_true':[]
    }).append(pd.get_dummies(predict_df[['outlook','windy']])).fillna(0).astype(int)
    predict_df = predict_df[['temperature','humidity']].join(predict_one_hot_encoded_df)


    # Reload the trained and persisted model.
    golf_dt_clf = DecisionTreeClassifierModel.load_model(persist_dir=PERSISTENCE_DIR, model_id=GOLF_MODEL_1)
    
    # Run the prediction.
    predicted_label = golf_dt_clf.predict(X_test=predict_df)


    return JSONResponse(content={
        'status':'ok',
        'prediction':predicted_label[0]
    }, status_code=200)

