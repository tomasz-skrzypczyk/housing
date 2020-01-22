import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import pandas
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit

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

    housing = strat_train_set.copy()


    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    from sklearn.pipeline import Pipeline
    from sklearn_features.transformers import DataFrameSelector

    cat_attribs = ["view"]
    cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs))])
    #Pearson coefficient
    # corr_matrix = housing.corr()
    # print(corr_matrix["price"]).sort_values(ascending=False)