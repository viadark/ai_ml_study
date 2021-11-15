import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from io import StringIO
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
    return x, y

def read_model_result(file):
    print(f"model result file path : {file}")
    f = open(file, 'r')
    coef = list(map(float, f.readline().rstrip().split()))
    inter = float(f.readline().rstrip())
    f.close()
    return coef, inter

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--answer_data',
        type=str,
        help="Input answer data csv"
    )
    argument_parser.add_argument(
        '--trained_coef',
        type=str,
        help="Train result coef"
    )
    argument_parser.add_argument(
        '--trained_intercept',
        type=str,
        help="Train result intercept"
    )

    args = argument_parser.parse_args()
    boston = args.answer_data
    boston = load_data(boston)
    x, y = get_train_test_data(boston)
    scaler = MinMaxScaler()
    scaler.fit(x)
    X_scaled = scaler.transform(x)
    model = LinearRegression()
    model.coef_ = np.array(list(map(float, args.trained_coef.rstrip().split())))
    model.intercept_ = float(args.trained_intercept.rstrip())
    predict = model.predict(X_scaled)
    print(predict)
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y, predict)
    print(f'\nMSE on Answer data : {np.sqrt(mse)}')

    print()
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('linear_regression', LinearRegression())
    ])
    pipeline.fit(x, y)
    with open('/model.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)
    # how to upload "model.pkl" to AI Platform model