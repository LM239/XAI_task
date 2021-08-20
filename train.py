from sklearn.model_selection import train_test_split

from data.aux_data import best_features
from dataloader import load_csv, clean_data
from regressor import DTRegressor

if __name__ == "__main__":
    modelClass = DTRegressor()

    data = load_csv()
    data = clean_data(data)

    x = data[best_features]
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    modelClass.train_model(X_train, y_train)
    
    print("RMSE test: ", modelClass.evaluate_model(X_test, y_test, options={"squared": False}))
    print("RMSE train: ", modelClass.evaluate_model(X_train, y_train, options={"squared": False}))

    print("Feature importances:")
    sorted_importances = sorted([t for t in zip(x.columns, modelClass.get_feature_importance())], key=lambda x: x[1], reverse=True)
    for f_i in sorted_importances:
        print(f_i)

    modelClass.save_model("models/finished_model")
