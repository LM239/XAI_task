from data.aux_data import default_query, best_features, cat_mapping
from regressor import DTRegressor
from flask import Flask, request
import pandas as pd
import os

app = Flask(__name__)

modelClass = DTRegressor()
modelClass.load_model("finished_model.p")


def clean_queries(queries):
    df_data = {}
    for key in best_features + ['saledate', 'YearMade']:
        df_data[key] = [query[key] if key in query.keys() else None for query in queries]

    df = pd.DataFrame(data=df_data)
    df['saledate'] = df['saledate'].apply(pd.to_datetime)

    df["yearsold"] = df["saledate"].dt.year
    df["ageWhenSold"] = df["saledate"].dt.year - df["YearMade"]

    df = df[best_features]
    for col in best_features:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: cat_mapping[col][x] if x in cat_mapping[col] else default_query[col])
    return df


@app.post("/prediction")
def predict():
    if request.is_json and 'queries' in request.json:
        try:
            clean_data = clean_queries(request.json['queries'])
            preds = modelClass.predict(clean_data)
            return {"predictions": list(preds)}, 201
        except KeyError:
            return {"error": "Request is missing fields"}, 400
        except Exception as e:
            return {"error": "Unknown error"}, 500
    return {"error": "Request must be JSON"}, 415


if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)))
