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
import requests

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


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--answer_data',
        type=str,
        help="Input answer data csv"
    )

    args = argument_parser.parse_args()
    boston = args.answer_data
    boston = load_data(boston)
    x, y = get_train_test_data(boston)

    model = pickle.load("/data/model.pkl")
    
    predict = model.predict(x)
    print(predict)
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(y, predict)
    print(f'\nMSE on Answer data : {np.sqrt(mse)}')

    # how to upload "model.pkl" to AI Platform model
    # files = {
    #     'file': args.model,
    #     'name': "boston_linear"
    # }
    # res = requests.post(url="http://viadark.iptime.org:8888/models", files=files)
    # print(res)
    # print(res.text)