import pandas as pd
import argparse

def preprocess_data(data):
    data['ZN'] = data['ZN'].fillna(0)
    data['NOX'] = data['NOX'].fillna(data.NOX.mean())
    data.drop(['RAD'], axis='columns', inplace=True)
    data.drop(['CHAS'], axis='columns', inplace=True)
    return data

def run():
    data = pd.read_csv("data.csv")
    data = preprocess_data(data)
    print(data)