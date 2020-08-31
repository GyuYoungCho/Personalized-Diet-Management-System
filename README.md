# 건강한 식생활을 위한 영양 분석 시스템
- 주제: 현대인들의 가장 큰 관심사이자 고민인 건강 문제를 채식으로 해결해줄 수 있는 서비스를 제공하는 시스템입니다. 
- contributors: 조규영, 전선영, 이소현, 김나연, 강성빈


## 서비스 요약 설명
### 주제 선정 배경
- 국내 채식 인구가 10년간 10배 이상 증가함에 따라 채식주의자 관련 서비스에 대한 수요와 필요성이 증가하고 있다.  채식주의자가 채식을 하는 이유 중 1위가 건강이고, 채식주의자 이외의 국민들도 건강에 대한 관심이 커지고 있다. 큰 관심에 비해 영양소 부족을 겪고 있는 사람들이 많기 때문에 건강을 관리하고자 하는 사람들에게 편리한 서비스를 제공하고자 하였다. 

### 서비스 이용 방법
- 사용자가 먹은 음식 사진을 찍으면 서비스가 음식을 보고 어떤 음식인지 인식하게 된다. 그 음식의 영양 성분 데이터를 활용해서 영양성분을 누적하여 영양성분 권장 섭취량과 비교하여 부족한 영양소를 포함한  음식을 추천해준다. 
<img width="500" alt="시스템 구성도(2)" src="https://user-images.githubusercontent.com/49351511/91661773-f254f400-ea92-11ea-985f-756054092899.png">

## 시스템 구현 과정
- 시스템 구현 과정
<img width="500" alt="eda" src="https://user-images.githubusercontent.com/49351511/91689459-04797580-eb19-11ea-8431-ce389bf0fe5c.png">

1. 이미지 데이터 수집: image crawling 파일 코드를 통해 구글, 네이버, 다음 등에서 이미지 크롤링
2. 이미지 데이터 저장: SQLite3와 함께 database에 저장
3. 분석/모델링: Yolov3와 PyTorch, Python을 통해 모델 구축

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Yolov3 Detect 학습 결과

<img width="600" alt="시스템 구성도(2)" src="https://user-images.githubusercontent.com/49351511/91691866-3e4c7b00-eb1d-11ea-852a-fb45c1824c17.png">

  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Yolov3 Detect 결과 예시

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![최종4](https://user-images.githubusercontent.com/49351511/91690806-6dfa8380-eb1b-11ea-8f2c-db0622ad4d43.jpg)

4. 표현: Flask를 통해 사용자가 웹 어플리케이션으로 시스템을 사용할 수 있도록 함

- 음식 사진 업로드 페이지

<img width="700" alt="플라스크_2" src="https://user-images.githubusercontent.com/49351511/91731991-758e4c80-eb5c-11ea-8499-8f8e2a822bdd.png">

- 음식 추천 페이지

<img width="700" alt="플라스크_1" src="https://user-images.githubusercontent.com/49351511/91731825-39f38280-eb5c-11ea-9eb8-abd05c022e15.png">



## Installation

- Yolov3

1. 구글 코랩에서 drive mount 한다.
2. My drive에 !git clone https://github.com/ultralytics/yolov3.git
3. data 폴더 안에 custom 폴더를 만들고 그 안에 labels와 images폴더를 생성한다.
4. data를 8:2 비율로 나누고 data/custom/images/이미지 이름 형식으로 train.txt와 valid.txt를 작성하고 custom 폴더 안에 넣는다.
5. custom 폴더 안에 image의 class들을 index에 맞춰 작성하여 classes.names를 생성하고, class 개수와 train.txt와 valid.txt의 경로를 적은 custom.data파일을 생성한다.
6. images안에 수집한 이미지 데이터를 넣어준다. 
7. !wget https://pjreddie.com/media/files/darknet53.conv.74 weights 폴더 안에 다운 받고 처음 학습은 darknet53.conv.74로 시작
8. 이후 epoch 돌면서 .pt파일 저장되면 last.pt로 이어서 학습시키기

[출처] https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data
