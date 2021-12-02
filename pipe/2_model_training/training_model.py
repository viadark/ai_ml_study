import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from io import StringIO
import copy
from sklearn.pipeline import Pipeline
import pickle

def load_data(data):
    d = StringIO(data)
    boston = pd.read_csv(d)
    print(boston.shape)
    return boston

def get_train_test_data(boston):
    # preprocessing and several data
    total_cols = len(boston.columns)
    x = boston.iloc[:,:-1].values.reshape(-1,total_cols - 1)
    y = boston.iloc[:,-1]
    X_train, X_test, y_train, y_test  = train_test_split(x, y, test_size=0.3)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--data',
        type=str,
        help="Input data csv"
    )

    args = argument_parser.parse_args()
    boston = args.data
    boston = load_data(boston)

    X_train, X_test, y_train, y_test = get_train_test_data(boston)

    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('linear_regression', LinearRegression())
    ])
    print("make pipeline")
    pipeline.fit(X_train, y_train)
    print("fit complete")
    predict = pipeline.predict(X_test)
    print("predict complete")

    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, predict)
    print(f"mse : {np.sqrt(mse)}")

    with open('/data/model.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    print("pickle dump complete")
