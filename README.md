# ml-golf-demo-app
A demo MLOps project where a decision tree classifier is used to decide if you should play golf.
You can interact with the API manually (see below), or you could create another web scraper/data collection app that will validate the data before sending it to ```/train```.

The code can be repurposed for other ML projects. If you want to understand how it works, you can start with ```app.py```.

## Try it out

### Set up

Clone the repo

    git clone https://github.com/pugad/ml-golf-demo-app 
    cd ml-golf-demo-app 

Spin up the container

    docker-compose up -d --build

Create a virtual environment and install requests

    python -m venv venv
    .\venv\Scripts\activate
    pip install -U requests

### Interact with the API (manually)

Train the model

    python .\manualtests\train_manual.py localhost:8000/train

Request for a prediction

    python .\manualtests\predict_manual.py localhost:8000/predict
 
You can edit the JSON payloads in the *manualtests* directory.

    # In VSCode,
    code .\manualtests\manual_train_data.json
    code .\manualtests\manual_predict_data.json
