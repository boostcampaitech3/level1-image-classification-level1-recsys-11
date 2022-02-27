import shutil, os
import pandas as pd
from itertools import chain

def change_folder_name(df, idx):  
    id_ = df['error_id'][idx]    

    before_fold = [os.path.join(folder_dir, folder) for folder in folder_name if id_ in folder][0]
    shutil.move(before_fold, before_fold.replace(df['before'][idx], df['after'][idx]))

def change_file_name(df, idx):
    id_ = df['error_id'][idx]
    before_ = df['before'][idx]
    after_ = df['after'][idx]
    before_fold = [os.path.join(folder_dir, file) for file in all_file_list if id_ in file and (before_ in file or after_ in file)]

    tmp_src, tmp_dst = before_fold[1].split('.')
    tmp = tmp_src+'_tmp.'+tmp_dst

    os.rename(before_fold[0], tmp)
    os.rename(before_fold[1], before_fold[0])
    os.rename(tmp, before_fold[1])


############### 입력 ###############
print('복사를 원하는 경로를 넣어주세요. ex) /opt/ml/input/data')
data_tree = input()
print('입력하신 경로의 데이터가 복사될 경로를 넣어주세요. ex) /opt/ml/input/raw_data')
copy_tree = input()

print('데이터 바꾸기를 작업하실 경로를 지정해주세요. ex) /opt/ml/input/data')
folder_dir = input()+'/train/images'

print('바꿔야하는 데이터가 담긴 파일의 경로와 이름을 지정해주세요. ex) /opt/ml/level1-image-classification-level1-recsys-11/Data_Processing/change_table.csv')
need_change_file = input() 
####################################

shutil.copytree(data_tree, copy_tree)  # data_tree의 데이터를 copy_tree로 복사

folder_name_tmp = os.listdir(folder_dir)
folder_name = list(set([x.replace('._','') for x in folder_name_tmp]))  # '._' 제거
folder_name.sort()

file_list = [[folder_name[i]+"/"+file for file in os.listdir(os.path.join(folder_dir, folder_name[i])) if '._' not in file and 'ipynb' not in file] for i in range(len(folder_name))]
all_file_list = list(chain(*file_list)) # 전체 파일명 (폴더별로 가져와 2차원 -> 1차원으로 풀기)


df = pd.read_csv(need_change_file)
print('have to change {} file!'.format(len(df)))

print("="*30)

for idx in range(len(df)):
    try:
        line = df.iloc[idx]
        if line['type'] == 'file':
            # change_file_name(df, idx)
            print("change file_name - {} [{} <-> {}]!".format(df['error_id'][idx], df['before'][idx], df['after'][idx]))
        elif line['type'] == 'all':
            change_folder_name(df, idx)
            print("change folder_name - {} [{} -> {}]!".format(df['error_id'][idx], df['before'][idx], df['after'][idx]))
    except:
        print("fail - ", idx)
        
print("="*30)