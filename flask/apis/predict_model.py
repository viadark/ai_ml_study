from flask.helpers import make_response
from flask_restful import Resource, reqparse
import os
import pandas as pd
import importlib
import pickle

class PredictModel(Resource):
    def post(self, model_name):
        parser = reqparse.RequestParser()
        parser.add_argument('data', action='append')
        args = parser.parse_args()
        data = pd.read_csv(args['data'])
        total_cols = len(data.columns)
        x = data.iloc[:,:-1].values.reshape(-1,total_cols - 1)
        print(f'input data : {data}')
        
        base_path = f'./model/{model_name}'
        
        model = pickle.load(open(f'{base_path}/model.pkl', 'rb'))
        ret = model.predict(x)
        print(ret)
        return ret