from flask.helpers import make_response
from flask_restful import Resource, reqparse
import os
import pandas as pd
import importlib

class Models(Resource):
    def get(self):
        folder = os.getcwd() + "/models/"
        ret = {"models": []}
        print(f"curr folder = {folder}")
        for filename in os.listdir(folder):
            filename_spl = filename.split(".")
            if len(filename_spl) == 2 and filename_spl[1] == "py" and filename_spl[0] != "list_models":
                ret["models"].append(filename_spl[0])
            print(filename)
        return ret
    def post(self):
        parser = reqparse.RequestParser()
        print('in Search')
        parser.add_argument('name', type=str, required=True, help='model name cannot be blank!')
        parser.add_argument('data', action='append')
        args = parser.parse_args()
        pd.DataFrame(args['data']).to_csv("data.csv", index=False)
        module = importlib.import_module("models."+args['name'])
        ret = module.run()
        print(ret)
        return ret
