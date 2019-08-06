import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn import datasets
import sys

from flask import Flask
from flask_restful import Api, Resource, reqparse
    
class IrisEstimator(Resource):
        
    def get(self):
        try:
            # 파라미터 파싱
            parser = reqparse.RequestParser()
            parser.add_argument('sepal_length', required=True, type=float, help='sepal_length is required')
            parser.add_argument('sepal_width',  required=True, type=float, help='sepal_width is required')
            parser.add_argument('petal_length', required=True, type=float, help='petal_length is required')
            parser.add_argument('petal_width',  required=True, type=float, help='petal_width is required')
            args = parser.parse_args()
            
            # 데이터 전처리
            features = [args['sepal_length'], args['sepal_width'], args['petal_length'], args['petal_width']]
            features = np.reshape(features, (1, 4))
            
            print('features:', features, file=sys.stderr) # 예측 전 데이터 출력하여 확인
            
            # 예측
            with deep_learning_graph.as_default():
                Y_pred = deep_learning_model.predict_classes(features)
            return {'species': target_names[Y_pred[0]]} # 얘측 결과 JSON형식 리턴
        
        except Exception as e:
            return {'error': str(e)}

def init():
    global deep_learning_model, deep_learning_graph, target_names
    deep_learning_model = load_model('iris_model.h5') # 저장된 모델 로딩
    deep_learning_graph = tf.get_default_graph()
    
    # species 이름 로딩 ex. ['setosa', 'versicolor', 'virginica']
    target_names = datasets.load_iris().target_names

app = Flask('My First AI App')
api = Api(app)
api.add_resource(IrisEstimator, '/predict')

if __name__ == '__main__':
    init() # 초기화
    app.run()    
##    app.run(host='https://flask02.herokuapp.com', port=8000, debug=True)
##    app.run(host='0.0.0.0', port=8000, debug=True) # 8000포트로 실행
