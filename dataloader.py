import re
import numpy as np
import pandas as pd

from best_features import best_features
from gdps import gdps


def load_csv(file=r"data\TrainAndValid.csv"):
    """
        Returns: dataframe of given csv file
        Input (optional): path to .csv file
    """
    try:
        df = pd.read_csv(file, parse_dates=["saledate"], low_memory=False)
    except FileNotFoundError:
        # fix anaconda wDir error for jupyter notebook (not present in pycharm)
        print("Used abs path")
        df = pd.read_csv(r"C:\Users\larsm\Documents\2NOrconsult\data\TrainAndValid.csv", parse_dates=["saledate"],
                         low_memory=False)
    return df


def convert_to_inches(val):
    """
        Returns: input length in inches (int)
        Input: string on the form x' y" or float('nan')
    """
    if val == "None or Unspecified" or val != val:
        return None
    else:
        feet, inches = val.split("' ")
        return 12 * int(feet) + int(inches[:-1])


def strip_inches(val):
    """
        Returns: input length in inches (int)
        Input: string on the form x inch or x"
    """
    return float(re.sub(r" inch|\"", "", val)) if val not in [None, "None or Unspecified"] and val == val else None


def clean_data(data: pd.DataFrame):
    """
        Returns: cleaned input dataframe with no nan values (changed to mode, median or 'None or Unspecified'), and new date, agewhensold and stategdp columns
        Input: dataframe with fields given in data/Data_Dictionary.xlsx
    """
    data["yearsold"] = data["saledate"].dt.year
    data["monthsold"] = data["saledate"].dt.month
    data["ageWhenSold"] = data["saledate"].dt.year - data["YearMade"] #TODO: handle incorrect age (when yearMade is 1000 or saledate is before yearmade")

    data["stateGDP"] = data["state"].apply(lambda x: gdps[x])
    data = data.drop(["SalesID", "MachineID", "saledate", "fiModelDesc", "fiBaseModel"], axis='columns')

    numeric_obj = ["Stick_Length", "Undercarriage_Pad_Width", "Tire_Size"]
    for col, obj in zip(data.columns, data.dtypes):
        if obj == "object" and col not in numeric_obj:
            data[col] = data[col].fillna("None or Unspecified") # remove Null values
            data[col] = pd.Categorical(data[col]).codes # change category strings to discrete values

    for col in numeric_obj: # these values are given as categorical but are actually numerical, thus we transform them
        if col == "Stick_Length":
            data["Stick_Length"] = data[col].apply(convert_to_inches).astype(np.float64)
        else:
            data[col] = data[col].apply(strip_inches).astype(np.float64)
        data[col] = data[col].fillna(data[col].median())

    data["MachineHoursCurrentMeter"] = data["MachineHoursCurrentMeter"].fillna(
        data["MachineHoursCurrentMeter"].median())
    data["auctioneerID"] = data["auctioneerID"].fillna(data["auctioneerID"].mode()[0])
    return data


def save_data(df: pd.DataFrame, path, index=False):
    df.to_csv(path, index=index)


if __name__ == "__main__":
    from sklearn.tree import DecisionTreeRegressor

    data = load_csv()
    data = clean_data(data)

    x = data[best_features]
    y = data["SalePrice"]

    x.to_csv('data/test_csv.csv', index=False)

    model = DecisionTreeRegressor()
    model.fit(x, y)

    print("R2-score: ", model.score(x, y))
    print("Feature importances")
    for feature in zip(x.columns, model.feature_importances_):
        print(feature)