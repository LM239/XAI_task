import numpy as np
from sklearn.model_selection import train_test_split

from best_features import best_features
from dataloader import load_csv, clean_data
from regressor import DTRegressor

if __name__ == "__main__":
    modelClass = DTRegressor()

    data = load_csv()
    data = clean_data(data)

    x = data[best_features]
    y = data["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    grid_params = {"max_depth": [6, 8, 10],
                   "min_samples_split": np.arange(2, 16, 2),
                   "min_samples_leaf": np.arange(1, 16, 2),
                   "criterion": ["mse", "friedman_mse"]}

    print("Best params:")
    print(modelClass.train_grid_search(x, y, grid_params))