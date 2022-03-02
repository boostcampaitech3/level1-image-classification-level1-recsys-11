# 😷 Mask Classification with gender and age
## Explanation
- 🔥 P-STAGE-1 (COMPETETION), NAVER BOOSTCAMP AI-TECH 
- 🍿 RECSYS-11; RECFLIX

## ✏️ What We Did(가제)
- [🔎 회의 노트 (외부 노션 링크)](https://recflix.notion.site/d4de596a7ca440829a08153fecc93aa4)
- [각자 발표 자료 정리 한 것 링크(가제)]()
- 








# Base Line Code Gyeongtae update version v.2.0.3

## 🔨 작업환경 구성하기
- clone repository
    ```bash
    git clone https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-11.git
    ```
- **필수**) 본인의 **branch**로 이동
    ```bash
    git checkout your_work_branch
    ```
- 필수 패키지 설치 (본인이 원하는 가상환경에서 작업하는 것을 추천합니다!, venv, conda ...)
    ```bash
    pip install -r ./level1-image-classification-level1-recsys-11/requirements.txt
    ```
- **BaseLineCodeV2** 디렉토리를 활용해서 작업하면 됩니다!!


## TO DO
- Layer에 Drop-Out을 사용하여 Regularization 방법을 사용한다. (일반화 성능을 올리기 위함)
- TTA 방법을 위한 TestDataset 구현

## 🔎 ISSUE
- Dataloader OOM
    - 해결방법 1  
    loader > **num_workers = 0** (train.py > train(), valid())
    - 해결방법 2   
    batch_size를 줄여나간다. 

## 🔎 업데이트 노트
<<<<<<< HEAD

### v.2.1.4
- **train.py**
    - (03.02) Class간 예측 정확도(Accuracy)를 확인하고, 이를 통해서 Weight를 부여하기 위한 작업을 하기 위해, train 중, Validation에서 Class간 정확도를 가시적으로 확인할 수 있는 로그를 만들었습니다.
    - 학습시 f1 score 수정  
        train 시에 임시로 학습과정에서 f1 score를 볼 수 있게 설정해놨는데, Batch size 단위로 f1 score로 보는 것은, forum에서도 언급된 바와 같이, 그 값의 신뢰성이 떨어집니다. 그래서 training 시 batch 단위의 f1 score를 폐기하고, validation의 경우에는 기존에 batch 단위로 기록되어 평균을 구하는 방식으로 기록하였었는데, 이 부분도, Metric의 정확한 의도 전달을 위해서 전체 validation set에 대한 f1 score를 기록하도록 하였습니다. 
    - 개편된 f1 score로 Best f1 score의 모델 파라미터 정보를 `best_f1.pth`로 기록할 수 있게 되었습니다. 

- **inference.py**
    - (argparser) `--state`  
        기존에는 학습된 모델을 acc metric 최고 성능 기준인 `best.pth`를 기준으로 추론하도록 설정되어있었지만, `best_f1.pth`, `last.pth` 중에서도 골라서 사용할 수 있도록 하였습니다. 
        - `best.pth`: acc 기준으로 최고 성능을 나타낸 state를 불러옵니다. 
        - `best_f1.pth`: f1(macro) 기준으로 최고성능을 나타낸 state를 불러옵니다.
        - `last.pth`: 마지막 epoch 학습의 state를 불러옵니다. 

### v.2.1.3

마스크 착용 여부(mask) 학습모델과 성별,나이 분류모델(genderAge) 을 앙상블하여 룰베이스로 output을 만드는 inference 코드 추가

**inference 기능 추가**

```bash
    #앙상블 이용 argument 추가 default값 False -> False시 기존의 inference 방식으로 작동
    python3 inference.py --isEnsemble True
```

- **inference.py**
    - (function) **`inference_by_single_models`**
    - (parameter) `(which_model, data_dir, model_dir, output_dir, args)`
        - which_model을 인자로 받아 mask 또는 genderAge 둘 중에 하나를 inference 함
        - output 디렉토리의 `output_mask.csv` 또는 `output_genderAge.csv` 파일로 결과 저장
         

    - (function) **`inference_ensemble`**
    - (parameter) `(data_dir, model_dir, output_dir, args)`
        - 마스크 분류 모델과 성별 나이 분류 모델을 앙상블하여 최종 제출 파일을 생성
        - `inference_by_single_models`을 이용하여 각 모델의 결과를 생성하고 그 결과를 이용하여 룰베이스로 최종 제출파일 생성 및 저장
        - output 디렉토리의 `output_ensemble.csv` 파일로 결과 저장
    

### v.2.1.2

모델 별 데이터 셋 추가 및 MLflow run user 추적기능 추가

**모델 별 데이터 셋 추가**
- **dataset.py**
    - (Dataset) **`MaskSplitByProfileDatasetForAlbumOnlyMask`**
        - 마스크 착용 여부 재 레이블링 하여 데이터를 피딩
         - **Class Description:**

                | Class | 마스크 착용 유형 | 세부 착용 유형 | Counts |
                | --- | --- | --- | --- |
                | 0 | Wear | Wear | 2700 X 5 |
                | 1 | Incorrect | nose mask |  |
                | 1 | Incorrect | mouse mask |  |
                | 2 | Not Wear | Not Wear | 2700 X 1 |

        <br>

    - (Dataset) **` MaskSplitByProfileDatasetForAlbumOnlyGenderAge`**
        - 성별, 나이 두가지 class를 조합하여 재 레이블링하여 데이터를 피딩
         - **Class Description:**
         
                | Class | Gender | Age | Counts |
                | --- | --- | --- | --- |
                | 0 | male | < 30 |  |
                | 1 | male | ≥ 30 and < 60 |  |
                | 2 | male | ≥ 60 |  |
                | 3 | female | < 30 |  |
                | 4 | female | ≥ 30 and < 60 |  |
                | 5 | female | ≥ 60 |  |
        <br>

**MLflow run user 추적기능 추가**
MLproject argument에 --user를 이용하여 user 추적 가능하도록 함
```bash
name: first_project

entry_points:
  main:
    command: "python3 train.py \
    # 유저 이름 추가
    --user kijung"
```
### v.2.1.1
- **model.py**
    - (Model) `Vgg11` 추가
    - (Model) `Vgg11Freeze` 추가 
    - (Model) `Vgg13` 이름변경 (원래 Vgg13bn)
    - (Model) `Vgg13Freeze` 이름변경 (원래 Vgg13bnFreeze)
    - (Model) `Vgg16` 추가
    - (Model) `Vgg16Freeze` 추가 
    - (Model) `Inception` 추가
        - input size is MUST **299x299**
    - (Model) `ResNet18Dropout` 추가
        - 기존의 `ResNet18` 모델에서 마지막 FC-layer에 값이 전달되기 전에 Dropout을 달아봤습니다. 
        - `__init__`에서 (fc) layer에 register_forward_hook을 이용하여, 모델을 직접 print해도 구조에 포함되지는 않습니다. 

    - (Model) `ResNet18MSD` 추가
        - ResNet18 Mulit Sample Dropout
        - ResNet18의 구조를 그대로 사용하고, 5개의 Dropout으로 ResNet의 결과값에 적용하고 Weight를 공유하는 Linear layer에 넣어서 나온 결과값의 평균을 결과값으로 사용하는 모델입니다.
        - 5개의 dropout layer와 1개의 linear layer로 구성되어있으며, 각 dropout의 percentile = 0.5로 설정했습니다.
        - 자세한 내용은 다음 캐글 Discussion에서 참고했습니다. [8th Place Solution (4 models simple avg) - Qishen Ha | Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100961)
    - 2022.03.01) 가중치 초기화 작업이 마지막 레이어에 적용되지 않고, ResNet 밑단을 초기화했음.        
    - (Model) `EfficientNetB3` 추가
        - image size : 300 (recommended)
    - 과적합 방지를 위해서 일부 pre-trained layer를 Freeze해줬습니다.
        - (Model) `ResNet18FreezeTop6` 추가
        - (Model) `ResNet34FreezeTop6` 추가
        - (Model) `EfficientNetB0FreezeTop3` 추가
            - image size : 224 (recommended)
        - (Model) `EfficientNetB4FreezeTop3` 추가
            - image size : 380 (recommended)
    

- **dataset.py**
    - `AugForInception`
        - 가운데 이미지 사이즈를 299x299로 Crop하고
        - 각종 인기있는 Albumentation 기능을 사용하여 Augmentation을 만들었습니다. 
        - 만들어진 목표는 `Inception` 모델을 더 잘 학습시키기 위해서 사용했지만, 현재(2.27) 만들어진 모든 Aug 기법에서 가장 좋은 성능을 보여줍니다.

    - `CustomAlbumentationAug` (Example Aug for module)
        - Albumentation 에 있는 transform 기능을 사용하고 싶다면 해당 Augmentation 포맷을 사용하여 제작할 수 있습니다. 

    - `MaskSplitByProfileDatasetForAlbum`
        - MaskSplitByProfileDataset을 통해서 train하고 싶고, Albumentation 모듈로 transform을 사용하는 경우 사용해야할때 사용하는 Dataset입니다.
        - Albumentation으로 Transform하는 Augmentation을 사용할 경우 해당 Dataset을 사용해주세요

    - `TestDatasetForAlbum` 
        - 나중에 TTA method 를 구현하기 위해서 미리 Albumentation 전용 TestDataset 모듈을 추가했습니다. 
        - 해당 모듈을 통해서 Inference 시에 Albumentation의 Augmentation기능을 수행할 수 있습니다.
### v.2.1.0
MLflow로 원격로깅이 가능해졌습니다.

MLflow 설치  
```bash
pip3 install mlflow
```

원격 로깅 주소  
http://101.101.210.160:30001


**mlflow 로깅 사용법**

MLflow 로깅 방법은 노션 docs에 있으니 참고  
https://www.notion.so/recflix/mlflow-f6390e20a43e474eb5c5ab01ec4c36cf

**원격서버에 실험 추가하기**

`experiment_name` 변수에 원하는 실험이름을 정하면 mlflow에 실험등록  
이미 존재하면 해당 실험에 로그가 저장됨

- **train.py**

```python
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        remote_server_uri ='http://101.101.210.160:30001'
        mlflow.set_tracking_uri(remote_server_uri)
        experiment_name = "/my-experiment-log2remote"
```

베이스라인코드에서 MLflow를 실행하기 위한 MLproject 파일 추가

- **MLproject**

`entry_points`의 `main`에 실행 argument 입력 후 저장
```
name: first_project

entry_points:
  main:
    command: "python3 train.py \
    --optimizer Adam \
    --epochs 3 \
    --criterion cross_entropy \
    --model BaseModel \
    --batch_size 128 \
    --name test_mlflow_server"
```

**MLflow를 이용한 실험 시작**  
MLproject에 있는 디렉토리에서 다음의 명령어로 학습 및 로깅 수행
```bash
mlflow run -e main . --no-conda
```

### v.2.0.3
이제 Albumentation을 활용해서 Augmentation을 사용할 수 있습니다!🎉🎉🎉  
- **dataset.py**
    - (transform) `GoodAugmentation` 
        - Albumentation
        - 여러가지 많이 사용하는 Augmentation를 무작위로 집어넣었음. (실험용)
        - 개인적으로 실험해 볼 수 있도록 example guide도 제공하였음.
    
    - (Dataset) **`MaskBaseDatasetForAlbum`**
        - Albumentation 모듈을 활용하여 Augmentation 기법을 사용하기 위해서는 특별한 데이터셋이 필요하다.
        - 해당 데이터셋은 `MaskBaseDataset`의 Child-Class이다.
        - 요청시 `TestDataset`, `MaskSplitByProfileDataset` 기반도 만들어드림. 

- **model.py**
  - (Mode) `Vgg13Bn` 추가
  - (Mode) `Vgg13BnFreeze` 추가


### v.2.0.2
- **model.py**
    - import torch 추가
    - (Model) `ResNet18Freeze` 추가

    - (Model) `EfficientNetB3` 추가
    - (Model) `EfficientNetB3Freeze` 추가

    - (Model) `ViTPretrainedFreeze` 추가
        - **Method** : Feature Extraction(Freeze Pretrained Convolutional Layer)
        - **size** : **MUST** (386,386)
        - **issue** : Calculation Validation 에서 Out Of Memory 현상이 발생된다. (aistage server 기준) 해결 방법은, validation batch size를 작게 조절해야한다. `--valid_batch_size 64`를 추천한다.
            - 이미지 사이즈가 (386, 386) 고정인 부분과, train, validation, inference 시 모든 상황에서 batch_size를 64로 설정해줘야하는것은 굉장한 단점.
            - **학습속도가 매우느리다.** 
            - pretrained-freeze 모델, LearningRate = 0.001로 학습시 기존의 학습보다 나은 성능을 보여준다.
            - TO DO ) Freeze를 풀고, LearningRate = 0.0001로 다시 설정 후 비교해본다. 
        - **실험 변수 정보**
            ```bash
            python train.py --epochs 5 --dataset MaskBaseDataset --augmentation BaseAugmentation --batch_size 64 --model ViTPretrainedFreeze --optimizer Adam --name 'VitPretrained' --resize 384 384 --valid_batch_size 64 
            ```
        - **실험 결과 정보**
            ```
            [Val] acc : 85.6878%, loss: 0.4184, f1 score ['macro'] : 0.7261   || best acc : 85.6878%, best loss: 0.4184, best f1 score: 0.737 
            LB : f1 0.5607, acc 62.5556
            ```
        - reference: https://github.com/lukemelas/PyTorch-Pretrained-ViT#loading-pretrained-models
    - ~~(Mode) `Vgg13Bn` 추가~~
    - ~~(Mode) `Vgg13BnFreeze` 추가~~


- **train.py**
    - argparser로 resize하는 방법 변경(*inference.py*도 같이 변경)
        ```python
        parser.add_argument("--resize", nargs="+",type=int, default=[128, 96], help='resize size for image')  
        ```
        `--resize 128 96` 이런식으로 입력하면 사이즈를 변경할 수 있게됩니다. 
    - 학습 결과 Log에서 F1 score를 확인할 수 있게되었음. (정확한 방법은 아니지만, 가늠하기 위해서 사용하면 된다. 참고 자료: [discuss.pytorch.org::Calculating F1 score over batched data](https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348))
-  **dataset.py**
    - Augmentation (Transform) 기능 추가
    - `MyAugmentationCenter`
        - (256, 256) 크기의 제일 중간 이미지 Crop
    - `MyAugmentationBust` 
        - (384, 384) size의 상반신 이미지 Crop
    
### v.2.0.1 
- **dataset.py**
    - **dataset.py** > `MaskBaseDataset` update
        > `image_paths`, `mask_labels`, `gender_labels`, `age_labels` 를 init 내부에 선언해줘서, path가 누적되는 현상을 해소해준다. 


### v.2.0.0 
updated at 02.24 from Boostcamp ai tech

---
# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`
