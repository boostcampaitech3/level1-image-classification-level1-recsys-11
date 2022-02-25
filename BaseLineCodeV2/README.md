# Base Line Code Gyeongtae update version v.2.0.0


### v.2.0.1 - dataset.py
- 내용
    - dataset.py > MaskBaseDataset update
        > `image_paths`, `mask_labels`, `gender_labels`, `age_labels` 를 init 내부에 선언해줘서, path가 누적되는 현상을 해소해준다. 


### v.2.0.0 
update at 02.24 by Boostcamp ai tech

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
