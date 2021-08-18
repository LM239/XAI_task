# app.py
import sys

from data.best_features import default_features
from dataloader import clean_data
from regressor import MachineRegressor
from flask import Flask, request
import pandas as pd

app = Flask(__name__)

modelClass = MachineRegressor()
modelClass.load_model("models/finished_model")

default_query = {'ageWhenSold': [9.0], 'yearsold': [2006.0], 'UsageBand': [3], 'fiSecondaryDesc': [108], 'fiModelSeries': [108], 'fiModelDescriptor': [92], 'ProductSize': [5], 'fiProductClassDesc': [1], 'state': [8], 'ProductGroup': [3], 'ProductGroupDesc': [3], 'Drive_System': [3], 'Enclosure': [5], 'Forks': [0], 'Pad_Type': [1], 'Ride_Control': [1], 'Stick': [1], 'Transmission': [4], 'Turbocharged': [0], 'Blade_Extension': [0], 'Blade_Width': [5], 'Enclosure_Type': [2], 'Engine_Horsepower': [1], 'Hydraulics': [0], 'Pushblock': [0], 'Ripper': [1], 'Scarifier': [0], 'Tip_Control': [0], 'Coupler': [2], 'Coupler_System': [0], 'Grouser_Tracks': [0], 'Hydraulics_Flow': [1], 'Track_Type': [0], 'Thumb': [2], 'Pattern_Changer': [1], 'Grouser_Type': [1], 'Backhoe_Mounting': [0], 'Blade_Type': [4], 'Travel_Controls': [5], 'Differential_Type': [3], 'Steering_Controls': [4], 'Stick_Length': [116.0], 'Undercarriage_Pad_Width': [28.0], 'Tire_Size': [20.5], 'MachineHoursCurrentMeter': [0.0], 'auctioneerID': [1.0]}

def cleanQueries(queries):
    df = pd.DataFrame(data={key: [query[key] if key in query.keys() and query[key] not in [None, float('nan')] else default_query[key] for query in queries] for key in default_features})
    df['saledate'] = df['saledate'].apply(pd.to_datetime)

    df["yearsold"] = df["saledate"].dt.year
    df["ageWhenSold"] = df["saledate"].dt.year - df["YearMade"]


@app.post("/predict")
def add_country():
    if request.is_json and 'queries' in request.json:
        try:
            cleanQueries(request.json['queries'])
            return {"prediction": 420}, 201
        except KeyError as e:
            print(e)
            return {"error": "Request failed"}, 400
        except Exception as e:
            print(e)
            return {"error": "Internal error"}, 500
    return {"error": "Request must be JSON"}, 415


