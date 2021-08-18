from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


class MachineRegressor:
    def __init__(self):
        self.model = None

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def train_model(self, x, y):
        self.model = DecisionTreeRegressor(min_samples_split=12, min_samples_leaf=9, max_depth=9, criterion='friedman_mse')
        self.model.fit(x, y)

    def train_grid_search(self, x, y, grid):
        grid = RandomizedSearchCV(DecisionTreeRegressor(), n_jobs=8, param_distributions=grid, n_iter=100, verbose=True, cv=10)
        grid.fit(x, y)
        return grid.best_params_

    def evaluate_model(self, test_x, test_y, metric=mean_squared_error, options=None):
        if options is None:
            options = {}
        if self.model:
            preds = self.model.predict(test_x)
            return metric(test_y, preds, **options)
        return None

    def get_feature_importance(self):
        if self.model:
            return list(self.model.feature_importances_)
        return None