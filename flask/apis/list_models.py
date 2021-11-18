from flask.helpers import make_response
from flask_restful import Resource, reqparse
import os
import pandas as pd
import importlib
import errno
import werkzeug

class Models(Resource):
    def get(self):
        folder = os.getcwd() + "/models/"
        ret = {"models": []}
        print(f"curr folder = {folder}")
        for filename in os.listdir(folder):
            filename_spl = filename.split(".")
            print(filename)
            ret["models"].append(filename)
        return ret
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('name', type=str)
        parser.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        parser.add_argument('scaler', type=werkzeug.datastructures.FileStorage, location='files')
        args = parser.parse_args()
        target_name = args['name']
        print(args['name'])
        #target_name = "boston_model"
        print(__file__)
        print(os.path.realpath(__file__))
        print(os.path.abspath(__file__))
        print(os.path.dirname(__file__))
        base_path = f"{os.path.dirname(__file__)}/../models/{target_name}/"
        #realpath = os.path.abspath(__file__).split("/")
        #base_path = f"./models/{target_name}/"
        if not os.path.exists(os.path.dirname(base_path)):
            try:
                os.makedirs(os.path.dirname(base_path))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        with open(f'{base_path}/model.pkl', "wb") as f:
            f.write(args['file'].read())
        
        if args['scaler'] != None:
            with open(f'{base_path}/scaler.pkl', "wb") as f:
                f.write(args['scaler'].read())
        else:
            print("scaler is null")
        
        print('model saved')
        return 'model saved'
