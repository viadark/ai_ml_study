import pandas as pd
import argparse

def preprocess_data(data):
    data['ZN'] = data['ZN'].fillna(0)
    data['NOX'] = data['NOX'].fillna(data.NOX.mean())
    data.drop(['RAD'], axis='columns', inplace=True)
    data.drop(['CHAS'], axis='columns', inplace=True)
    data.drop(['RAD'], axis='columns', inplace=True)
    data.drop(['CHAS'], axis='columns', inplace=True)
    data = data.drop(index=range(335,len(data)))
    return data

if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        '--data_path', type=str,
        help="Input data path"
    )

    args = argument_parser.parse_args()
    data = pd.read_csv(args.data_path)
    print(data.shape)
    print(f"load data : {args.data_path}")
    data = preprocess_data(data)

    data.to_csv(args.data_path[1:], index=False)