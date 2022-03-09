

## 🔎 업데이트 노트
<<<<<<< HEAD

## v.1.2.0
- **face_cropping_top.ipynb**
    - SSD object detector를 활용하여 도출한 상반신 영역 crop (incorrect, normal 이미지)
    - `face_cropping_top()`
        1. 모든 파일 경로를 리스트로 불러온다.
        2. SSD object detector로 상반신 영역을 탐지하고, 해당 부분을 저장한다.
        3. fail의 경우 이전 값으로 크롭한다.

    - `face_cropping_top_avg()`
        1. 모든 파일 경로를 리스트로 불러온다.
        2. 평균값에 해당하는 부분을 크롭해서 저장한다.

    - **코드 실행**   
    모든 데이터 탐지를 진행하기에는 메모리나 시간적인 부분 모두 비효율적입니다.   
    그래서 평균 값으로 크롭하는 함수를 만들어두었으니 그 함수를 사용해서 한 번에 바꾸시는 것을 추천드립니다.   
    -> 탐지가 되지 않는 fail data를 평균으로 크롭하는 테스트 결과 대부분 상반신 부분을 알맞게 크롭함을 확인했습니다.   
    활용은 부족한 데이터 증강용으로 normal과 incorrect만 사용하시는건 어떨까합니다.


## v.1.1.1
- **change_filename.ipynb**
    - os.rename() -> shutil.move()로 변경했습니다.

## v.1.1.0
- **change_filename.ipynb**
    - 미스라벨링된 데이터 리스트, 이를 변경하는 코드 추가   
    - **코드 실행 순서**
        1. 지정한 경로로 데이터를 복사한다. (raw data 저장용)
        2. 작업 경로와 바꿔야하는 데이터를 받는다.
        > 작업 경로는 학습이 진행되는 경로인 "/opt/ml/input/data"로 지정하시는 것을 권장합니다.
        3. 파일 내용대로 작업을 지정한 경로에 파일 변경이 일어난다.
            - 인적사항(성별, 나이 등)이 잘못된 경우 -> 폴더명이 변경됩니다.
            - 파일의 라벨링(마스크 착용여부)이 잘못된 경우 -> 파일명이 변경됩니다.

    - **코드 활용 방법**
    1. 코드 실행 - 코드가 있는 곳에서 실행하면 입력해야할 내용들이 순차적으로 뜹니다.   
    `python change_mislabeling.py`
    2. 순차적으로 입력
        > 라벨링을 또 바꾸어야할 경우 재사용을 위해서 경로를 입력으로 받도록 만들었습니다. 처음 실행시 아래와 같이 넣으시면 될거고, 이후에는 상황에 맞게 경로와 파일 입력해주시면 됩니다.
        - 입력1) 복사를 원하는 경로를 넣습니다.   
        `/opt/ml/input/data`
        - 입력2) 데이터가 복사될 경로를 넣습니다.
        `/opt/ml/input/raw_data`
        - 입력3) 데이터 바꾸기를 작업할 경로를 넣습니다.
        `/opt/ml/input/data`
        - 입력4) 바꿀 내용이 있는 파일의 경로를 넣습니다.
        `/opt/ml/level1-image-classification-level1-recsys-11/Data_Processing/change_table.csv`
    3. 입력을 완료하시면 변경되어야할 id 개수가 출력되고, 작업이 완료되면 파일/폴더 변경사항이 출력됩니다. 오류가 뜨는 경우 fail과 함께 id가 출력될 것이므로 확인해주시면 됩니다.

- **age_gender_detection.ipynb**   
    - Opencv의 하르분류기 활용한 age&gender detection 코드 추가