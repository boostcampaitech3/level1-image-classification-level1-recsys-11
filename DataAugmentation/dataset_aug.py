import shutil, os
import pandas as pd
from itertools import chain
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2

'''
dataset.py에서 MaskBaseDataset 부분 다음과 같이 수정

class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = { # TODO
        "mask1": MaskLabels.MASK, "rmask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK, "rmask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK, "rmask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK, "rmask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK, "rmask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT, "rincorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL, "rnormal": MaskLabels.NORMAL,
    }


이 파이썬 파일 실행하면 data_copies라고 백업파일 형성되고 data 디렉터리에는 회전한 사진이 추가됩니다.
'''









# os.mkdir('/opt/ml/input/data_invert')
# os.mkdir('/opt/ml/input/data_invert/train')
# new_dir = '/opt/ml/input/data_invert/train/images'
# os.mkdir(new_dir)
shutil.copytree('/opt/ml/input/data', '/opt/ml/input/data_copies')

def invert_image(img_path, folder_dir, folder_name, _file_name, ext):
    image = Image.open(img_path)
    image = np.array(image)
    invert_image = np.zeros((512, 384, 3), dtype=np.uint8)
    invert_image[0:256, 0:256] = [255, 255, 255]
    invert_image = invert_image - image
    invert_image = Image.fromarray(invert_image)
    invert_image.save(folder_dir + '/' + folder_name + '/' + 'i' + _file_name + ext)


def rotate_image(img_path, folder_dir, folder_name, _file_name, ext):
    img = cv2.imread(img_path)
    rows,cols = img.shape[0:2]

    #---① 회전을 위한 변환 행렬 구하기
    # 회전축:중앙, 각도:45, 배율:1
    m = cv2.getRotationMatrix2D((cols/2,rows/2),4,1)  

    #---② 변환 행렬 적용
    imgs = cv2.warpAffine(img, m,(cols, rows))
    imgs = imgs[0:512,0:384]
    cv2.imwrite(folder_dir + '/' + folder_name + '/' + 'r' + _file_name + ext, imgs)


def horizontalflip_image(img_path, folder_dir, folder_name, _file_name, ext):
    image = Image.open(img_path)
    image = np.array(image)
    flip_image = np.flip(image, axis=1)
    flip_image = Image.fromarray(flip_image)
    flip_image.save(folder_dir + '/' + folder_name + '/' + 'f' + _file_name + ext)


folder_dir = '/opt/ml/input/data/train/images' # 폴더 디렉토리
folder_name_tmp = os.listdir(folder_dir) # 폴더 이름 목록

for folder_name in tqdm(folder_name_tmp):
    if folder_name.startswith('.'):
        continue

    img_folder = os.path.join(folder_dir, folder_name) # 폴더 디렉토리

    for file_name in os.listdir(img_folder):
        _file_name, ext = os.path.splitext(file_name)
        if _file_name.startswith('.'):  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
            continue

        img_path = os.path.join(folder_dir, folder_name, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
        
        rotate_image(img_path, folder_dir, folder_name, _file_name, ext)
        # horizontalflip_image(img_path, folder_dir, folder_name, _file_name, ext)