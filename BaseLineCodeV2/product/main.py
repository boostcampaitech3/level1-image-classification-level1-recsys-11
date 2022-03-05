import flask
from flask import Flask, request, render_template
# from sklearn.externals import joblib
import numpy as np
from scipy import misc

# from BaseLineCodeV2.inference import load_model
# from ..BaseLineCodeV2.inference import load_model

app = Flask(__name__)

# Main page routing 
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')

#
# @app.route('/predict', methods=['POST'])
# def make_prediction():
#     if request.method == 'POST':

#         # 업로드 파일 분기
#         file = request.files['image']
#         if not file : return render_template('index.html', label = 'No Files')

#         # 이미지 픽셀 정보 읽기
#         img = misc.imread(file)
#         img = img[:, :, :3]
#         img = img.reshape(1,-1)

        # load_model

        # # prediction = model.predict(img)
        # model.eval()
        # pred = model(img)
        # pred = pred.argmax(dim=-1)
        # pred = pred.cpu().numpy()

if __name__ == '__main__':
    # 모델 로드
    # ml/model.py 선 실행 후 생성
    # model = joblib.load('./model/model.pkl')
    # Flask 서비스 스타트
    app.run(host='0.0.0.0', port=30002, debug=True)
