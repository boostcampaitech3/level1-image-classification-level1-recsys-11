# π· Mask Classification with gender and age
## Explanation
- π₯ P-STAGE-1 (COMPETETION), NAVER BOOSTCAMP AI-TECH 
- πΏ RECSYS-11; RECFLIX

## βοΈ What We Did(κ°μ )
- [π νμ λΈνΈ (μΈλΆ λΈμ λ§ν¬)](https://recflix.notion.site/d4de596a7ca440829a08153fecc93aa4)
- [κ°μ λ°ν μλ£ μ λ¦¬ ν κ² λ§ν¬(κ°μ )]()
- 








# Base Line Code Gyeongtae update version v.2.0.3

## π¨ μμνκ²½ κ΅¬μ±νκΈ°
- clone repository
    ```bash
    git clone https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-11.git
    ```
- **νμ**) λ³ΈμΈμ **branch**λ‘ μ΄λ
    ```bash
    git checkout your_work_branch
    ```
- νμ ν¨ν€μ§ μ€μΉ (λ³ΈμΈμ΄ μνλ κ°μνκ²½μμ μμνλ κ²μ μΆμ²ν©λλ€!, venv, conda ...)
    ```bash
    pip install -r ./level1-image-classification-level1-recsys-11/requirements.txt
    ```
- **BaseLineCodeV2** λλ ν λ¦¬λ₯Ό νμ©ν΄μ μμνλ©΄ λ©λλ€!!


## TO DO
- Layerμ Drop-Outμ μ¬μ©νμ¬ Regularization λ°©λ²μ μ¬μ©νλ€. (μΌλ°ν μ±λ₯μ μ¬λ¦¬κΈ° μν¨)
- TTA λ°©λ²μ μν TestDataset κ΅¬ν

## π ISSUE
- Dataloader OOM
    - ν΄κ²°λ°©λ² 1  
    loader > **num_workers = 0** (train.py > train(), valid())
    - ν΄κ²°λ°©λ² 2   
    batch_sizeλ₯Ό μ€μ¬λκ°λ€. 

## π μλ°μ΄νΈ λΈνΈ

### v.2.2.0
NNI(AutoML) νμ©μ ν΅ν νμ΄νΌνλΌλ―Έν° νλμ΄ κ°λ₯ν©λλ€.

NNI μ€μΉ
```
git clone -b v2.6 https://github.com/Microsoft/nni.git
pip install nni
```

**NNI μ¬μ©λ²**
nni μ€νμ μν΄μλ 'config.yml' νμΌκ³Ό 'search_space.json'νμΌμ΄ νμν©λλ€.

1. config.yml
- MLflow μ¬μ© μ νμ©νλ MLProject νμΌκ³Ό λΉμ·νλ€κ³  μκ°νμλ©΄ λ©λλ€.
- `searchSpaceFile:`μλ νμν  νμ΄νΌνλΌλ―Έν°κ° μλ νμΌμ μ§μ ν©λλ€.   
 'search_space.json' νμΌμ λ£μ΄μ£Όμλ©΄ λ©λλ€.
- `trialCommand:`μλ mlflowμ entry_pointsμ κ°μ΄ μλ ₯νμλ©΄ λ©λλ€!   
μ λ μ λ κ²½λ‘λ‘ λ£μμ΅λλ€.
- 'trialConcurrency'μλ λͺ κ°μ λͺ¨λΈμ λλ¦΄μ§ λ£μ΄μ£Όμλ©΄ λ©λλ€.
```
searchSpaceFile: search_space.json
trialCommand: python train.py \
    --experiment general \
    --dataset MaskSplitByProfileDatasetForAlbum \
    --seed 42 \
    --epochs 10 \
    --resize 300 300 \
    --model EfficientNetB3 \
    --valid_batch_size 472 \
    --name nni_model_experiment \
    --user your_name

trialGpuNumber: 1
trialConcurrency: 2

tuner:
  name: TPE
  classArgs:
    optimize_mode: maximize
trainingService:
  platform: local
  useActiveGpu: true
```
2. search_space.json
- νμμ μνλ νμ΄νΌνλΌλ―Έν°μ λ²μλ₯Ό μ§μ ν΄μ£Όμλ©΄ λ©λλ€.
```
{
    "batch_size": {"_type":"choice", "_value": [16, 32, 64, 128]},
    "optimizer": {"_type":"choice", "_value": ["Adam", "SGD", "AdamW"]},
    "augmentation": {"_type":"choice", "_value": ["myAlbumentationAug", "AugForInception"]},   
    "lr":{"_type":"choice","_value":[0.00001, 0.0001, 0.001]}
}
```

**nniλ₯Ό μ΄μ©ν μ€ν μμ**  
BaseLineCodeV2 λλ ν λ¦¬μμ λ€μμ λͺλ Ήμ΄λ‘ νμ΅ μν
```bash
nnictl create --config AutoML/config_mask.yml --port 8080
```
λμ€λ λ§ν¬ "http://127.0.0.1:8080/" λ₯Ό ν΄λ¦­νμλ©΄ νμ΅ νΈλνΉμ΄ κ°λ₯ν©λλ€.

μ μ§νμ€ λλ μλμ λͺλ Ήμ΄λ₯Ό μ¬μ©ν΄μ£ΌμΈμ.
```bash
nnictl stop --all
```
### v.2.1.5
- **train.py**
    - earlystopping κΈ°λ₯ μΆκ°
    - earlystopping.pyμμ λͺ¨λ λΆλ¬μ μλνλ λ°©μμλλ€. validation lossκ° 5μν­ μ΄μ κ°μ κ²½μ° λ©μΆλλ‘ μ€μ νμμ΅λλ€.

### v.2.1.4
- **train.py**
    - (03.02) Classκ° μμΈ‘ μ νλ(Accuracy)λ₯Ό νμΈνκ³ , μ΄λ₯Ό ν΅ν΄μ Weightλ₯Ό λΆμ¬νκΈ° μν μμμ νκΈ° μν΄, train μ€, Validationμμ Classκ° μ νλλ₯Ό κ°μμ μΌλ‘ νμΈν  μ μλ λ‘κ·Έλ₯Ό λ§λ€μμ΅λλ€.
    - νμ΅μ f1 score μμ   
        train μμ μμλ‘ νμ΅κ³Όμ μμ f1 scoreλ₯Ό λ³Ό μ μκ² μ€μ ν΄λ¨λλ°, Batch size λ¨μλ‘ f1 scoreλ‘ λ³΄λ κ²μ, forumμμλ μΈκΈλ λ°μ κ°μ΄, κ·Έ κ°μ μ λ’°μ±μ΄ λ¨μ΄μ§λλ€. κ·Έλμ training μ batch λ¨μμ f1 scoreλ₯Ό νκΈ°νκ³ , validationμ κ²½μ°μλ κΈ°μ‘΄μ batch λ¨μλ‘ κΈ°λ‘λμ΄ νκ· μ κ΅¬νλ λ°©μμΌλ‘ κΈ°λ‘νμμλλ°, μ΄ λΆλΆλ, Metricμ μ νν μλ μ λ¬μ μν΄μ μ μ²΄ validation setμ λν f1 scoreλ₯Ό κΈ°λ‘νλλ‘ νμμ΅λλ€. 
    - κ°νΈλ f1 scoreλ‘ Best f1 scoreμ λͺ¨λΈ νλΌλ―Έν° μ λ³΄λ₯Ό `best_f1.pth`λ‘ κΈ°λ‘ν  μ μκ² λμμ΅λλ€. 

- **inference.py**
    - (argparser) `--state`  
        κΈ°μ‘΄μλ νμ΅λ λͺ¨λΈμ acc metric μ΅κ³  μ±λ₯ κΈ°μ€μΈ `best.pth`λ₯Ό κΈ°μ€μΌλ‘ μΆλ‘ νλλ‘ μ€μ λμ΄μμμ§λ§, `best_f1.pth`, `last.pth` μ€μμλ κ³¨λΌμ μ¬μ©ν  μ μλλ‘ νμμ΅λλ€. 
        - `best.pth`: acc κΈ°μ€μΌλ‘ μ΅κ³  μ±λ₯μ λνλΈ stateλ₯Ό λΆλ¬μ΅λλ€. 
        - `best_f1.pth`: f1(macro) κΈ°μ€μΌλ‘ μ΅κ³ μ±λ₯μ λνλΈ stateλ₯Ό λΆλ¬μ΅λλ€.
        - `last.pth`: λ§μ§λ§ epoch νμ΅μ stateλ₯Ό λΆλ¬μ΅λλ€. 

### v.2.1.3

λ§μ€ν¬ μ°©μ© μ¬λΆ(mask) νμ΅λͺ¨λΈκ³Ό μ±λ³,λμ΄ λΆλ₯λͺ¨λΈ(genderAge) μ μμλΈνμ¬ λ£°λ² μ΄μ€λ‘ outputμ λ§λλ inference μ½λ μΆκ°

**inference κΈ°λ₯ μΆκ°**

```bash
    #μμλΈ μ΄μ© argument μΆκ° defaultκ° False -> Falseμ κΈ°μ‘΄μ inference λ°©μμΌλ‘ μλ
    python3 inference.py --isEnsemble True
```

- **inference.py**
    - (function) **`inference_by_single_models`**
    - (parameter) `(which_model, data_dir, model_dir, output_dir, args)`
        - which_modelμ μΈμλ‘ λ°μ mask λλ genderAge λ μ€μ νλλ₯Ό inference ν¨
        - output λλ ν λ¦¬μ `output_mask.csv` λλ `output_genderAge.csv` νμΌλ‘ κ²°κ³Ό μ μ₯
         

    - (function) **`inference_ensemble`**
    - (parameter) `(data_dir, model_dir, output_dir, args)`
        - λ§μ€ν¬ λΆλ₯ λͺ¨λΈκ³Ό μ±λ³ λμ΄ λΆλ₯ λͺ¨λΈμ μμλΈνμ¬ μ΅μ’ μ μΆ νμΌμ μμ±
        - `inference_by_single_models`μ μ΄μ©νμ¬ κ° λͺ¨λΈμ κ²°κ³Όλ₯Ό μμ±νκ³  κ·Έ κ²°κ³Όλ₯Ό μ΄μ©νμ¬ λ£°λ² μ΄μ€λ‘ μ΅μ’ μ μΆνμΌ μμ± λ° μ μ₯
        - output λλ ν λ¦¬μ `output_ensemble.csv` νμΌλ‘ κ²°κ³Ό μ μ₯
    

### v.2.1.2

λͺ¨λΈ λ³ λ°μ΄ν° μ μΆκ° λ° MLflow run user μΆμ κΈ°λ₯ μΆκ°

**λͺ¨λΈ λ³ λ°μ΄ν° μ μΆκ°**
- **dataset.py**
    - (Dataset) **`MaskSplitByProfileDatasetForAlbumOnlyMask`**
        - λ§μ€ν¬ μ°©μ© μ¬λΆ μ¬ λ μ΄λΈλ§ νμ¬ λ°μ΄ν°λ₯Ό νΌλ©
         - **Class Description:**

                | Class | λ§μ€ν¬ μ°©μ© μ ν | μΈλΆ μ°©μ© μ ν | Counts |
                | --- | --- | --- | --- |
                | 0 | Wear | Wear | 2700 X 5 |
                | 1 | Incorrect | nose mask |  |
                | 1 | Incorrect | mouse mask |  |
                | 2 | Not Wear | Not Wear | 2700 X 1 |

        <br>

    - (Dataset) **` MaskSplitByProfileDatasetForAlbumOnlyGenderAge`**
        - μ±λ³, λμ΄ λκ°μ§ classλ₯Ό μ‘°ν©νμ¬ μ¬ λ μ΄λΈλ§νμ¬ λ°μ΄ν°λ₯Ό νΌλ©
         - **Class Description:**
         
                | Class | Gender | Age | Counts |
                | --- | --- | --- | --- |
                | 0 | male | < 30 |  |
                | 1 | male | β₯ 30 and < 60 |  |
                | 2 | male | β₯ 60 |  |
                | 3 | female | < 30 |  |
                | 4 | female | β₯ 30 and < 60 |  |
                | 5 | female | β₯ 60 |  |
        <br>

**MLflow run user μΆμ κΈ°λ₯ μΆκ°**
MLproject argumentμ --userλ₯Ό μ΄μ©νμ¬ user μΆμ  κ°λ₯νλλ‘ ν¨
```bash
name: first_project

entry_points:
  main:
    command: "python3 train.py \
    # μ μ  μ΄λ¦ μΆκ°
    --user kijung"
```
### v.2.1.1
- **model.py**
    - (Model) `Vgg11` μΆκ°
    - (Model) `Vgg11Freeze` μΆκ° 
    - (Model) `Vgg13` μ΄λ¦λ³κ²½ (μλ Vgg13bn)
    - (Model) `Vgg13Freeze` μ΄λ¦λ³κ²½ (μλ Vgg13bnFreeze)
    - (Model) `Vgg16` μΆκ°
    - (Model) `Vgg16Freeze` μΆκ° 
    - (Model) `Inception` μΆκ°
        - input size is MUST **299x299**
    - (Model) `ResNet18Dropout` μΆκ°
        - κΈ°μ‘΄μ `ResNet18` λͺ¨λΈμμ λ§μ§λ§ FC-layerμ κ°μ΄ μ λ¬λκΈ° μ μ Dropoutμ λ¬μλ΄€μ΅λλ€. 
        - `__init__`μμ (fc) layerμ register_forward_hookμ μ΄μ©νμ¬, λͺ¨λΈμ μ§μ  printν΄λ κ΅¬μ‘°μ ν¬ν¨λμ§λ μμ΅λλ€. 

    - (Model) `ResNet18MSD` μΆκ°
        - ResNet18 Mulit Sample Dropout
        - ResNet18μ κ΅¬μ‘°λ₯Ό κ·Έλλ‘ μ¬μ©νκ³ , 5κ°μ DropoutμΌλ‘ ResNetμ κ²°κ³Όκ°μ μ μ©νκ³  Weightλ₯Ό κ³΅μ νλ Linear layerμ λ£μ΄μ λμ¨ κ²°κ³Όκ°μ νκ· μ κ²°κ³Όκ°μΌλ‘ μ¬μ©νλ λͺ¨λΈμλλ€.
        - 5κ°μ dropout layerμ 1κ°μ linear layerλ‘ κ΅¬μ±λμ΄μμΌλ©°, κ° dropoutμ percentile = 0.5λ‘ μ€μ νμ΅λλ€.
        - μμΈν λ΄μ©μ λ€μ μΊκΈ Discussionμμ μ°Έκ³ νμ΅λλ€. [8th Place Solution (4 models simple avg) - Qishen Ha | Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100961)
    - 2022.03.01) κ°μ€μΉ μ΄κΈ°ν μμμ΄ λ§μ§λ§ λ μ΄μ΄μ μ μ©λμ§ μκ³ , ResNet λ°λ¨μ μ΄κΈ°ννμ.        
    - (Model) `EfficientNetB3` μΆκ°
        - image size : 300 (recommended)
    - κ³Όμ ν© λ°©μ§λ₯Ό μν΄μ μΌλΆ pre-trained layerλ₯Ό Freezeν΄μ€¬μ΅λλ€.
        - (Model) `ResNet18FreezeTop6` μΆκ°
        - (Model) `ResNet34FreezeTop6` μΆκ°
        - (Model) `EfficientNetB0FreezeTop3` μΆκ°
            - image size : 224 (recommended)
        - (Model) `EfficientNetB4FreezeTop3` μΆκ°
            - image size : 380 (recommended)
    

- **dataset.py**
    - `AugForInception`
        - κ°μ΄λ° μ΄λ―Έμ§ μ¬μ΄μ¦λ₯Ό 299x299λ‘ Cropνκ³ 
        - κ°μ’ μΈκΈ°μλ Albumentation κΈ°λ₯μ μ¬μ©νμ¬ Augmentationμ λ§λ€μμ΅λλ€. 
        - λ§λ€μ΄μ§ λͺ©νλ `Inception` λͺ¨λΈμ λ μ νμ΅μν€κΈ° μν΄μ μ¬μ©νμ§λ§, νμ¬(2.27) λ§λ€μ΄μ§ λͺ¨λ  Aug κΈ°λ²μμ κ°μ₯ μ’μ μ±λ₯μ λ³΄μ¬μ€λλ€.

    - `CustomAlbumentationAug` (Example Aug for module)
        - Albumentation μ μλ transform κΈ°λ₯μ μ¬μ©νκ³  μΆλ€λ©΄ ν΄λΉ Augmentation ν¬λ§·μ μ¬μ©νμ¬ μ μν  μ μμ΅λλ€. 

    - `MaskSplitByProfileDatasetForAlbum`
        - MaskSplitByProfileDatasetμ ν΅ν΄μ trainνκ³  μΆκ³ , Albumentation λͺ¨λλ‘ transformμ μ¬μ©νλ κ²½μ° μ¬μ©ν΄μΌν λ μ¬μ©νλ Datasetμλλ€.
        - AlbumentationμΌλ‘ Transformνλ Augmentationμ μ¬μ©ν  κ²½μ° ν΄λΉ Datasetμ μ¬μ©ν΄μ£ΌμΈμ

    - `TestDatasetForAlbum` 
        - λμ€μ TTA method λ₯Ό κ΅¬ννκΈ° μν΄μ λ―Έλ¦¬ Albumentation μ μ© TestDataset λͺ¨λμ μΆκ°νμ΅λλ€. 
        - ν΄λΉ λͺ¨λμ ν΅ν΄μ Inference μμ Albumentationμ AugmentationκΈ°λ₯μ μνν  μ μμ΅λλ€.
### v.2.1.0
MLflowλ‘ μκ²©λ‘κΉμ΄ κ°λ₯ν΄μ‘μ΅λλ€.

MLflow μ€μΉ  
```bash
pip3 install mlflow
```

μκ²© λ‘κΉ μ£Όμ  
http://101.101.210.160:30001


**mlflow λ‘κΉ μ¬μ©λ²**

MLflow λ‘κΉ λ°©λ²μ λΈμ docsμ μμΌλ μ°Έκ³   
https://www.notion.so/recflix/mlflow-f6390e20a43e474eb5c5ab01ec4c36cf

**μκ²©μλ²μ μ€ν μΆκ°νκΈ°**

`experiment_name` λ³μμ μνλ μ€νμ΄λ¦μ μ νλ©΄ mlflowμ μ€νλ±λ‘  
μ΄λ―Έ μ‘΄μ¬νλ©΄ ν΄λΉ μ€νμ λ‘κ·Έκ° μ μ₯λ¨

- **train.py**

```python
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        remote_server_uri ='http://101.101.210.160:30001'
        mlflow.set_tracking_uri(remote_server_uri)
        experiment_name = "/my-experiment-log2remote"
```

λ² μ΄μ€λΌμΈμ½λμμ MLflowλ₯Ό μ€ννκΈ° μν MLproject νμΌ μΆκ°

- **MLproject**

`entry_points`μ `main`μ μ€ν argument μλ ₯ ν μ μ₯
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

**MLflowλ₯Ό μ΄μ©ν μ€ν μμ**  
MLprojectμ μλ λλ ν λ¦¬μμ λ€μμ λͺλ Ήμ΄λ‘ νμ΅ λ° λ‘κΉ μν
```bash
mlflow run -e main . --no-conda
```

### v.2.0.3
μ΄μ  Albumentationμ νμ©ν΄μ Augmentationμ μ¬μ©ν  μ μμ΅λλ€!πππ  
- **dataset.py**
    - (transform) `GoodAugmentation` 
        - Albumentation
        - μ¬λ¬κ°μ§ λ§μ΄ μ¬μ©νλ Augmentationλ₯Ό λ¬΄μμλ‘ μ§μ΄λ£μμ. (μ€νμ©)
        - κ°μΈμ μΌλ‘ μ€νν΄ λ³Ό μ μλλ‘ example guideλ μ κ³΅νμμ.
    
    - (Dataset) **`MaskBaseDatasetForAlbum`**
        - Albumentation λͺ¨λμ νμ©νμ¬ Augmentation κΈ°λ²μ μ¬μ©νκΈ° μν΄μλ νΉλ³ν λ°μ΄ν°μμ΄ νμνλ€.
        - ν΄λΉ λ°μ΄ν°μμ `MaskBaseDataset`μ Child-Classμ΄λ€.
        - μμ²­μ `TestDataset`, `MaskSplitByProfileDataset` κΈ°λ°λ λ§λ€μ΄λλ¦Ό. 

- **model.py**
  - (Mode) `Vgg13Bn` μΆκ°
  - (Mode) `Vgg13BnFreeze` μΆκ°


### v.2.0.2
- **model.py**
    - import torch μΆκ°
    - (Model) `ResNet18Freeze` μΆκ°

    - (Model) `EfficientNetB3` μΆκ°
    - (Model) `EfficientNetB3Freeze` μΆκ°

    - (Model) `ViTPretrainedFreeze` μΆκ°
        - **Method** : Feature Extraction(Freeze Pretrained Convolutional Layer)
        - **size** : **MUST** (386,386)
        - **issue** : Calculation Validation μμ Out Of Memory νμμ΄ λ°μλλ€. (aistage server κΈ°μ€) ν΄κ²° λ°©λ²μ, validation batch sizeλ₯Ό μκ² μ‘°μ ν΄μΌνλ€. `--valid_batch_size 64`λ₯Ό μΆμ²νλ€.
            - μ΄λ―Έμ§ μ¬μ΄μ¦κ° (386, 386) κ³ μ μΈ λΆλΆκ³Ό, train, validation, inference μ λͺ¨λ  μν©μμ batch_sizeλ₯Ό 64λ‘ μ€μ ν΄μ€μΌνλκ²μ κ΅μ₯ν λ¨μ .
            - **νμ΅μλκ° λ§€μ°λλ¦¬λ€.** 
            - pretrained-freeze λͺ¨λΈ, LearningRate = 0.001λ‘ νμ΅μ κΈ°μ‘΄μ νμ΅λ³΄λ€ λμ μ±λ₯μ λ³΄μ¬μ€λ€.
            - TO DO ) Freezeλ₯Ό νκ³ , LearningRate = 0.0001λ‘ λ€μ μ€μ  ν λΉκ΅ν΄λ³Έλ€. 
        - **μ€ν λ³μ μ λ³΄**
            ```bash
            python train.py --epochs 5 --dataset MaskBaseDataset --augmentation BaseAugmentation --batch_size 64 --model ViTPretrainedFreeze --optimizer Adam --name 'VitPretrained' --resize 384 384 --valid_batch_size 64 
            ```
        - **μ€ν κ²°κ³Ό μ λ³΄**
            ```
            [Val] acc : 85.6878%, loss: 0.4184, f1 score ['macro'] : 0.7261   || best acc : 85.6878%, best loss: 0.4184, best f1 score: 0.737 
            LB : f1 0.5607, acc 62.5556
            ```
        - reference: https://github.com/lukemelas/PyTorch-Pretrained-ViT#loading-pretrained-models
    - ~~(Mode) `Vgg13Bn` μΆκ°~~
    - ~~(Mode) `Vgg13BnFreeze` μΆκ°~~


- **train.py**
    - argparserλ‘ resizeνλ λ°©λ² λ³κ²½(*inference.py*λ κ°μ΄ λ³κ²½)
        ```python
        parser.add_argument("--resize", nargs="+",type=int, default=[128, 96], help='resize size for image')  
        ```
        `--resize 128 96` μ΄λ°μμΌλ‘ μλ ₯νλ©΄ μ¬μ΄μ¦λ₯Ό λ³κ²½ν  μ μκ²λ©λλ€. 
    - νμ΅ κ²°κ³Ό Logμμ F1 scoreλ₯Ό νμΈν  μ μκ²λμμ. (μ νν λ°©λ²μ μλμ§λ§, κ°λ νκΈ° μν΄μ μ¬μ©νλ©΄ λλ€. μ°Έκ³  μλ£: [discuss.pytorch.org::Calculating F1 score over batched data](https://discuss.pytorch.org/t/calculating-f1-score-over-batched-data/83348))
-  **dataset.py**
    - Augmentation (Transform) κΈ°λ₯ μΆκ°
    - `MyAugmentationCenter`
        - (256, 256) ν¬κΈ°μ μ μΌ μ€κ° μ΄λ―Έμ§ Crop
    - `MyAugmentationBust` 
        - (384, 384) sizeμ μλ°μ  μ΄λ―Έμ§ Crop
    
### v.2.0.1 
- **dataset.py**
    - **dataset.py** > `MaskBaseDataset` update
        > `image_paths`, `mask_labels`, `gender_labels`, `age_labels` λ₯Ό init λ΄λΆμ μ μΈν΄μ€μ, pathκ° λμ λλ νμμ ν΄μν΄μ€λ€. 


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
