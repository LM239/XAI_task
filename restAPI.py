from sklearn.model_selection import train_test_split

from best_features import best_features
from dataloader import load_csv, clean_data
from regressor import MachineRegressor

if __name__ == "__main__":
    modelClass = MachineRegressor()

    data = load_csv()
    data = clean_data(data)

    x = data[best_features]
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    modelClass.load_model("models/finished_model")

    print("Feature importances:")
    sorted_importances = sorted([t for t in zip(x.columns, modelClass.get_feature_importance())], key=lambda x: x[1], reverse=True)
    for f_i in sorted_importances:
        print(f_i)

    modelClass.save_model("models/finished_model2")