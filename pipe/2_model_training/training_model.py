import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from io import StringIO
import copy

def load_data(data):
    d = StringIO(data)
    boston = pd.read_csv(d)
    print(boston.shape)
    return boston

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

    final_model = LinearRegression()
    min_mse = 987654321.9
    for _ in range(10000):
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
        if min_mse > np.sqrt(mse):
            print(f'\nMSE on test data : {np.sqrt(mse)}')
            min_mse = np.sqrt(mse)
            final_model = copy.deepcopy(model)

    print(f"final coef : {final_model.coef_}")
    print(f"last model coef : {model.coef_}")
    f = open('/trained_coef', 'w')
    for c in final_model.coef_:
        f.write(f'{c} ')
    f.close()
    f = open('/trained_intercept', 'w')
    f.write(f'{final_model.intercept_}')
    f.close()