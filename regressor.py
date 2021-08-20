import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


class Regressor:
    """
        Parent class for a regression model
    """
    def __init__(self):
        self.model = None

    def load_model(self, path):
        """
            Input: path to stored in format supported by pickle
            Returns: None
        """
        self.model = pickle.load(open(path, 'rb'))

    def save_model(self, complete_path):
        """
            Input: path to save the self.model using pickle
            Returns: None
        """
        pickle.dump(self.model, open(complete_path, 'wb'))

    def train_model(self, x, y):
        """
            Input: features x and targets y
            Returns: None
        """
        pass

    def predict(self, x):
        """
            Input: features x to predict on
            Returns: model predictions for the input features
        """
        return self.model.predict(x) if self.model else None

    def evaluate_model(self, test_x, test_y, metric=mean_squared_error, options=None):
        """
            Input: features x and y to test on
                  Optional: metric for evalution (default: mean_squared_error), and options dict for said metric
            Returns: model evaluation on the selected metric
        """
        if options is None:
            options = {}
        if self.model:
            preds = self.model.predict(test_x)
            return metric(test_y, preds, **options)
        return None


class DTRegressor(Regressor):
    """
        Decision tree regressor class with options for Randomized hyperparameter search
    """
    def train_model(self, x, y):
        self.model = DecisionTreeRegressor(min_samples_split=12, min_samples_leaf=9, max_depth=9, criterion='friedman_mse')
        self.model.fit(x, y)

    def train_grid_search(self, x, y, grid, options=None):
        """
            Inputs: features x, targets y, grid parameters, and remaining RandomizedSearchCV options (optional)
            Returns: Best params tested
        """
        if options is None:
            options = {"n_jobs": 8,
                       "param_distributions": grid,
                       "verbose": True,
                       "n_iter": 100,
                       "cv": 10}
        grid = RandomizedSearchCV(DecisionTreeRegressor(), **options)
        grid.fit(x, y)
        return grid.best_params_

    def get_feature_importance(self):
        """
            Returns: feature importances for the decision tree
        """
        return self.model.feature_importances_ if self.model else None

