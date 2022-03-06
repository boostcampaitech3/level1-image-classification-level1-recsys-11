import os
import flask
from flask import Flask, request, render_template

import math
import numpy as np

import imageio as iio
from PIL import Image

import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop, InterpolationMode

# input image process
# resize = (300, 300)
mean=(0.548, 0.504, 0.479)
std=(0.237, 0.247, 0.246)
transforms = Compose([
            # CenterCrop(360),
            # # Resize(300, InterpolationMode), #, Image.BILINEAR
            # Resize(300, Image.BILINEAR), #, Image.BILINEAR            
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])


# model load
class EfficientNetB3MSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.dropout_p = 0.5
        self.dropout_n = 5
        self.efficientnet = models.efficientnet_b3(pretrained=False)

        self.dropouts = nn.ModuleList([
            nn.Dropout(self.dropout_p) for _ in range(self.dropout_n)
        ])
        
        in_features = self.efficientnet.classifier[1].out_features
        self.linear = nn.Linear(in_features=in_features, out_features=num_classes)

        # initialize w & b
        torch.nn.init.xavier_uniform_(self.linear.weight)
        stdv = 1 / math.sqrt(self.linear.in_features)
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = self.efficientnet(x)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.linear(dropout(x))
            else:
                h += self.linear(dropout(x))
        output = h / len(self.dropouts)
        return output

model = EfficientNetB3MSD(18)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model_path = os.path.join('./models/FocalEfficientNetB3MSD', 'best.pth' )
model.load_state_dict(torch.load(model_path, map_location=device))

# label checking 
Mask_type   = ['Wear', 'Incorrect', 'Not Wear']
Gender_type = ['Male', 'Female']
Age_type    = ['<30', '>=30 and <60', '>=60']

class_description = {i*6+j*3+k : f"{mask} & {gender} & {age}"  for i, mask in enumerate(Mask_type) for j, gender in enumerate(Gender_type) for k, age in enumerate(Age_type)}


app = Flask(__name__)

# Main page routing 
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':

        # 업로드 파일 분기
        # file = request.files['image']
        file = request.files['image']
        print('this is file:', file)
        if not file : return render_template('index.html', label = 'No Files')

        # 이미지 픽셀 정보 읽기
        # img = iio.imread(file)
        img = Image.open(file)
        image = transforms(img)    

        # prediction = model.predict(img)
        image = image.view(1,*image.size())
        model.eval()
        pred = model(image)
        pred = pred.argmax(dim=-1)
        pred = pred.cpu().numpy()

        label = class_description[pred[0]]


        result = label.split(' & ')
        mask_status = result[0]
        gender = result[1]
        age = result[2]

        return render_template('index.html', label = label, mask_status = mask_status, gender = gender, age = age)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=30002, debug= True)
