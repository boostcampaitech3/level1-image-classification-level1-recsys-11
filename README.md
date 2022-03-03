# level1-image-classification-level1-recsys-11
<div align="center">
  
  # ![14b2d48f3f2a607f89c00711328a4334](https://user-images.githubusercontent.com/58928739/156540522-8b4741e7-8113-4aaa-b1f0-81172cf1d40c.png)

  TEAM RECFLIX 11조  
  ## 마스크 착용 상태 분류 대회
</div>

# ✧ 프로젝트 주제

연령, 성별 마스크 착용 여부를 분류하는 모델을 개발한다.

# ✧ 프로젝트 개요

이미지를 데이터로 사용해 Gender(Male, Female), Age(<30, ≥30 and <60, ≥60), Mask(Wear, Incorrect, Corret)를 기준으로 총 18개의 클래스(ex Male, 20, Wear → Class0)로 분류하는 모델을 개발한다.

EDA 통해 훈련에 사용할 image와 라벨을 분석하여 데이터 불균형, 잘못된 라벨링을 탐색하고 이를 기준으로 데이터 Augmentation, 잘못된 라벨링의 수정을 통해 해결하고자 하였다. 또한 모델 선정에서는 이미지 분류에서 뛰어난 성적의 모델을 기준으로 다양한 실험을 통해 선정하였다. 그 외에도 Loss function, ensemble, learning rate, optimizer, 평가방법을 교육에서 배웠던 이론을 기반한 다양한 실험을 통해 선정했다.

# 📚 회의 노트 
[회의 노트 노션 링크](https://recflix.notion.site/d4de596a7ca440829a08153fecc93aa4)

# 🛠 BaseLineCodeV2
대회 진행에서 주어진 기본 베이스라인 코드를 개선해가며 저희만의 베이스라인 코드로 완성시켰습니다.  
추가된 기능과 개선사항은 BaseLineCodeV2의 README.md 를 참조 하시면 됩니다:)  
[베이스라인 코드](https://github.com/boostcampaitech3/level1-image-classification-level1-recsys-11/tree/main/BaseLineCodeV2)

# 👦🏻👩🏻 멤버 소개 
OOO : 역할<br> 
OOO : 역할<br> 
OOO : 역할<br> 
OOO : 역할<br> 
OOO : 역할<br> 

# 프로젝트 사용 툴
### python 3.8.5  
![Python-logo-notext](https://user-images.githubusercontent.com/58928739/156547814-abb34731-5ea9-4a02-8214-580f549e17c4.svg)
### pytorch 1.10.2
![488px-PyTorch_logo_black svg](https://user-images.githubusercontent.com/58928739/156548371-2e7044fc-273b-4f90-b8c9-6202c4af71c6.png)
### TensorBoard  
<img src="https://user-images.githubusercontent.com/58928739/156548790-734b199a-01bd-4499-b0ec-c79e82ba54ef.png" width="400" height="200"><br>  
### mlflow
<img src="https://user-images.githubusercontent.com/58928739/156549548-82a4e400-2b7c-41a7-8f39-41fb6fc85b1e.png" width="40%" height="40%"><br>
