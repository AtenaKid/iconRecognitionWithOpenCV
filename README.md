# iconRecognition
open cv HOG와 SVM을 이용한 아이콘 인식 알고리즘

1. 라벨이 정의된 아이콘 이미지 데이터를 수집(Positive data 17 클래스, Negative data 1 class)


2. HOG feature을 통해서 각 아이콘의gradient 정보를  추출하여 xml 파일로 저장


3. 각 아이콘 클래스의 특징을 하나의 Mat 데이터로 통합


4. SVM classifier을 이용하여 multi-class classification 모델을 train

6. 모델을 xml 파일로 저장

7. 다른 데이터로 모델을 test하여 성능 평가

