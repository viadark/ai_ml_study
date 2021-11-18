from flask.helpers import make_response
from flask_restful import Resource, reqparse
import os
import pandas as pd
import importlib
import pickle
import json
from flask import jsonify

class PredictModel(Resource):
    def post(self, model_name):
        print(f"predict_model at {model_name}")
        parser = reqparse.RequestParser()
        parser.add_argument('data', action='append')
        args = parser.parse_args()
        print(__file__)
        print(os.path.realpath(__file__))
        print(os.path.abspath(__file__))
        print(f"data : {args['data']}")
        base_path = f"{os.path.dirname(__file__)}/../models/{model_name}"
        print(base_path)
        data = pd.DataFrame(args['data'])
        x = data.iloc[:,:].values.reshape(-1,13)
        print(x)
        # total_cols = len(data)
        # x = data.iloc[:,:-1].values.reshape(-1,total_cols)
        # print(f"reshape data : {x}")
        scaler = pickle.load(open(f'{base_path}/scaler.pkl', 'rb'))
        model = pickle.load(open(f'{base_path}/model.pkl', 'rb'))
        x = scaler.transform(x)
        ret = model.predict(x)
        res = { "res" : str(ret)}
        print(ret)
        return json.dumps(res)