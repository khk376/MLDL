# 성적으로 클래스 구분하는 프로그램 만들어 보기
# 성적 : 국어점수, 영어점수
# 군집, 그룹 ... : 3개

'''
비지도 학습 개발 단계(의사 코드)
1단계 - library import
2단계 - 데이터는 제공받았다는 전제하에 데이터 확인
3단계 - 데이터 정제(육안. 데이터 많을경우에 전처리 API)
4단계 - 정제된 데이터를 read(pandas 사용)
5단계 - 몇개의 군집으로 개발할 것인지 설정(3)
6단계 - 학습시키기(fit)
7단계 - 차트화 하기(mglearn 사용)
8단계 - 예측(predict)

'''

import mglearn
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans

# matplotlib로 한글 데이터 사용 가능하게 하는 설정
# os 내에 설치된 한글 또는 임의로 설치해서 사용 가능
font_location="c:/Windows/fonts/malgun.ttf"
font_name = font_manager.FontProperties(fname=font_location).get_name()
matplotlib.rc("font", family=font_name)

# 점수를 보유한 ex05~.csv 로 DataFrame 생성하기
data = pd.read_csv("ex05_academy1.csv")
print(data.iloc[:, 1:])

# 군지 모델 구성
kmeans = KMeans(n_clusters=3)
kmeans.fit(data.iloc[:, 1:])
