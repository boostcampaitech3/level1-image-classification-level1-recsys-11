### 탐색적데이터분석(EDA)

 사람이 마스크를 착용하였는지 판별할 수 있는 모델을 학습하는데 사용됩니다. 모든 데이터셋은 아시아인 남녀로 구성되어 있고, 나이는 20대부터 70대까지 다양하게 분포되어 있으며 간략한 통계는 다음과 같다.

- 전체 사람 명 수 : 4,500
- 한 사람당 사진의 개수 : 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]
- 이미지 크기 : 384,512

입력값  → 마스크 착용 사진, 미착용 사진, 이상하게 착용한 사진

결과값 → Gender(Male, Female), Age(<30, ≥30 and <60, ≥60), Mask(Wear, Incorrect, Correct)개 3개의 feature를 통해 image하나당 18개의 class(0 ~ 17)로 분류한다.

![56bd7d05-4eb8-4e3e-884d-18bd74dc4864](https://user-images.githubusercontent.com/48706951/156685772-08c22924-51a8-4589-9ef6-1b7eb151869d.png)

