from models.decision_tree_classifier import DecisionTreeClassifierModel
from models.datamodels import TrainDataModel, PredictDataModel

from use_cases.train_golf_model import initialize_train_persist

from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from datasets.wrapper import get_weather_data

import pandas as pd
import os
import uuid

# initialize FastAPI
APP = FastAPI()

# mount static and templates folders
APP.mount('/static', StaticFiles(directory='static'), name='static')
TEMPLATES = Jinja2Templates(directory='templates')

# specify the path to the initial training dataset
WEATHER_DATASET = 'datasets/weather_data.csv'

# for the sake of this demo, let's just make the models persist in a folder here
PERSISTENCE_DIR = 'persist_dir'
if not os.path.exists(PERSISTENCE_DIR):
    os.mkdir(PERSISTENCE_DIR)

# initialize model ID/s
GOLF_MODEL_1 = 'golf-mlmodel-' + str(uuid.uuid4())[:8]

# initialize, train, and persist the model
features_df, labels = get_weather_data(WEATHER_DATASET)
initialize_train_persist(
    model_id=GOLF_MODEL_1,
    features_df=features_df,
    labels=labels,
    persist_dir=PERSISTENCE_DIR
)

# TODO: perform initial model evaluation here
# <-code to evaluate trained models->


@APP.get("/")
async def home(request:Request):
    return TEMPLATES.TemplateResponse("index.html", context={'request':request}, status_code=200)

@APP.get("/healthz")
async def healthz():
    return JSONResponse(content={'status':'ok'}, status_code=status.HTTP_200_OK)

@APP.post("/train")
async def train(data:TrainDataModel):
    '''
    Trains the model using data from the outside
    '''

    # validate obtained training data
    new_train_df = data.to_dataframe()
    if not len(new_train_df):
        return JSONResponse(
            content={'status':'error','reason':'invalid/empty data'},
            status_code=status.HTTP_400_BAD_REQUEST
        )
    

    # the implementation below isn't scalable
    # but it will suffice in this example
    # since we know we are using a small dataset.
    # We'll likely have to switch to a random forest
    # classifier and use dask for much larger datasets

    # get features and perform any necessary one-hot encoding and cleaning
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
    # But for other models, reloading it here will be useful for partial fitting.
    golf_dt_clf = DecisionTreeClassifierModel.load_model(persist_dir=PERSISTENCE_DIR, model_id=GOLF_MODEL_1)

    # we reload the weather dataset and combine it with the new training dataset
    # since we're using a DecisionTreeClassifier
    current_features_df, current_labels = get_weather_data(WEATHER_DATASET)
    prev_feature_count = len(current_features_df)
    prev_labels_count = len(current_labels)
    current_features_df = current_features_df.append(new_features_df).reset_index(drop=True)
    current_labels.extend(new_labels)

    # retrain and persist the model.
    golf_dt_clf.train(X_train=current_features_df, y_train=current_labels)
    golf_dt_clf.persist(PERSISTENCE_DIR)

    return JSONResponse(content={
        'status':'model trained',
        'previous_features_count':prev_feature_count,
        'previous_labels_count':prev_labels_count,
        'new_features_count':len(current_features_df),
        'new_labels_count':len(current_labels)
    }, status_code=status.HTTP_200_OK)

@APP.post("/predict")
async def predict(data:PredictDataModel):


    # validate obtained training data
    predict_df = data.to_dataframe()
    if not len(predict_df):
        return JSONResponse(
            content={'status':'error','reason':'invalid/empty data'},
            status_code=status.HTTP_400_BAD_REQUEST
        )

    # perform one-hot encoding
    predict_one_hot_encoded_df = pd.DataFrame({
        'outlook_sunny':[],
        'outlook_rainy':[],
        'outlook_overcast':[],
        'windy_false':[],
        'windy_true':[]
    }).append(pd.get_dummies(predict_df[['outlook','windy']])).fillna(0).astype(int)
    predict_df = predict_df[['temperature','humidity']].join(predict_one_hot_encoded_df)


    # Reload the trained and persisted model
    golf_dt_clf = DecisionTreeClassifierModel.load_model(persist_dir=PERSISTENCE_DIR, model_id=GOLF_MODEL_1)
    
    # Run the prediction
    predicted_label = golf_dt_clf.predict(X_test=predict_df)


    return JSONResponse(content={
        'status':'ok',
        'prediction':predicted_label[0]
    }, status_code=status.HTTP_200_OK)

