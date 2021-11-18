from flask import Flask, jsonify, request
from flask_restful import Api
from flask_cors import CORS, cross_origin
from apis.list_models import Models
from apis.predict_model import PredictModel

app = Flask(__name__)
#CORS(app)
api = Api(app)

@app.route('/')
def board():
    return "Temporary main page"

api.add_resource(Models, '/models')
api.add_resource(PredictModel, '/models/<model_name>')

if __name__ == '__main__':
    app.run(debug=True, port='8888')
