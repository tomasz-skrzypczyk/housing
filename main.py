import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.ensemble import RandomForestRegressor

HOUSING_PATH = "house.csv"

def load_housing_data(csv_path):
    return pd.read_csv(csv_path, parse_dates=['date'])

if __name__ == '__main__':
    #Load data from the directory
    housing = load_housing_data(HOUSING_PATH)

    #Preview
    print(housing.head())

    #metadata info
    print(housing.info())

    #We can observe that there are no missing values.

    #basic statistics
    print(housing.describe())

    #price_bin
    print(housing["price_bin"].value_counts())

    #variables distributions
    # housing.hist(bins=50, figsize=(20,15))
    # plt.show()


    #bedrooms distribution
    print(housing["bedrooms"].value_counts())
    #33 bedrooms - too high

    #bathrooms
    print(housing["bathrooms"].value_counts())

    #bathrooms are not integer values!
    print(housing["bathrooms"].describe())

    #dates
    print(housing["date"].describe())

    # attributes = ["bathrooms","bedrooms","price","floors" ,"view" , "condition","price_bin"]
    # scatter_matrix(housing[attributes], figsize=(12,8))
    # plt.show()


    # housing.plot(kind="scatter", x="long", y="lat", alpha=0.1, c="price", cmap=plt.get_cmap("jet"), colorbar=True)
    # plt.show()

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=38)
    #equal distribution over waterfront - believed to be an importat feature
    for train_index, test_index in split.split(housing, housing["waterfront"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    housing = strat_train_set.drop(["date", "price", "price_bin", "id"], axis=1)
    housing_price_float_labels = strat_train_set["price"].copy()
    housing_price_cat_labels =  strat_train_set["price_bin"].copy()

    housing_test = strat_test_set.drop(["date", "price", "price_bin", "id"], axis=1)
    housing_test_price_float_labels = strat_test_set["price"].copy()
    housing_test_price_cat_labels =  strat_test_set["price_bin"].copy()

    housing_num  = housing.drop("view", axis=1)
    num_attribs = list(housing_num)
    cat_attribs = ["view"]

    cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                             ('cat_encoder', OneHotEncoder())])
    num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),
                             ('std_scaler', StandardScaler())])

    full_pipeline = FeatureUnion(transformer_list=[('num_pipeline', num_pipeline),
                                                   ('cat_pipeline', cat_pipeline)])

    housing_prepared = full_pipeline.fit_transform(housing)
    housing_test_prepared = full_pipeline.fit(housing_test)




    from sklearn.linear_model import LinearRegression

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_price_float_labels)

    from sklearn.metrics import mean_squared_error
    import numpy as np

    # housing_float_predictions = lin_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_price_float_labels, housing_float_predictions)
    # tree_mse = np.sqrt(tree_mse)
    # print("Linear regression model loss", tree_mse)
    #bardzo słabo

    from sklearn.tree import DecisionTreeRegressor
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_price_float_labels)

    # housing_float_predictions = tree_reg.predict(housing_prepared)
    # tree_mse = mean_squared_error(housing_price_float_labels, housing_float_predictions)
    # tree_mse = np.sqrt(tree_mse)
    # print("Tree regression model loss", tree_mse)

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(tree_reg, housing_prepared, housing_price_float_labels, scoring='neg_mean_squared_error', cv=10)
    tree_rsme_scores = np.sqrt(-scores)


    def display_scores(scores):
        print("Results: ", scores)
        print("Mean: ", scores.mean())
        print("Standard deviation: ", scores.std())

    display_scores(tree_rsme_scores)




    lin_scores = cross_val_score(lin_reg, housing_prepared, housing_price_float_labels, scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores = np.sqrt(-lin_scores)
    print("Linear regression model loss")
    display_scores(lin_rmse_scores)

    #Pearson coefficient
    # corr_matrix = housing.corr()
    # print(corr_matrix["price"]).sort_values(ascending=False)

    #Random Forest models

    forest_reg = RandomForestRegressor()
    #source: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator=forest_reg, param_distributions=random_grid, n_iter=30, cv=3, verbose=2, random_state=42,  n_jobs=-1)
    #fit the random search model
    rf_random.fit(housing_prepared, housing_price_float_labels)
    print(rf_random.best_params_)

    print(rf_random.cv_results_)

    from sklearn.externals import joblib
    joblib.dump(rf_random.best_estimator_, "best_forest_estimator")
