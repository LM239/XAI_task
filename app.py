# app.py
from data.best_features import best_features
from regressor import MachineRegressor
from flask import Flask, request
import pandas as pd

app = Flask(__name__)

modelClass = MachineRegressor()
modelClass.load_model("models/finished_model")

def cleanQueries(queries):
    new_qs = [{key: query[key] for key in best_features if key not in ["ageWhenSold", "yearsold"]} for query in queries]
    print(new_qs)

@app.post("/predict")
def add_country():
    if request.is_json and 'queries' in request.json:
    try:
        cleanQueries(request.json['queries'])
        return {"prediction": 420}, 201
    except:
        return {"error": "Request failed"}, 400
    return {"error": "Request must be JSON"}, 415


