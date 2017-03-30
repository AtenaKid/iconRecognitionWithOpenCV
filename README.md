# iconRecognition
open cv HOG와 SVM을 이용한 아이콘 인식 알고리즘

1. 라벨이 정의된 아이콘 이미지 데이터를 수집(Positive data 8 클래스, Negative data 1 class)

2. 실제 분석하는 환경에서 아이콘은 다른 그림의 배경과 겹쳐 보일 수 있으므로, 배경을 합셩한 추가 데이터 생성

3. HOG 알고리즘을 통해서 각 아이콘의 edge feature를 추출하여 xml 데이터로 추출

4. 각 아이콘 클래스의 특징을 하나의 Mat 데이터로 통합

5. NU_SVM을 이용하여 multi-class classification 모델을 학습

6. 모델을 xml 데이터로 추출

7. 다른 데이터로 모델을 test하여 성능 평가

