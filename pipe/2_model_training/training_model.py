import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from io import StringIO

def load_data(data):
    d = StringIO(data)
    boston = pd.read_csv(d)
    print(boston.shape)
    return boston

def preprocess_data(data):
    data['ZN'] = data['ZN'].fillna(0)
    data['NOX'] = data['NOX'].fillna(data.NOX.mean())
    data.drop(['RAD'], axis='columns', inplace=True)
    data.drop(['CHAS'], axis='columns', inplace=True)
    return data

def reshape_data(data):
    total_cols = len(data.columns)
    x = data.iloc[:,:-1].values.reshape(-1,total_cols - 1)
    y = data.iloc[:,-1]
    return x, y

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
    boston = preprocess_data(boston)
    X_train, X_test, y_train, y_test = get_train_test_data(boston)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    predict = model.predict(X_test_scaled)
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y_test, predict)
    print(f'\nAccuracy Score on test data : {mse}')