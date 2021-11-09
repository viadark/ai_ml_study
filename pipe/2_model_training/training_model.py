import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from io import StringIO

def load_data(data):
    d = StringIO(data)
    boston = pd.read_csv(d)
    print(boston.shape)
    return boston


def get_train_test_data(boston):
    # preprocessing and several data
    train, test = train_test_split(boston, test_size=0.3)
    X_train = train
    X_test = test
    y_train = []
    y_test = []
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
    model = LinearRegression()
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print("accuracy print")