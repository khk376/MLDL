

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
data = pd.read_csv("ex05_academy2.csv")
print(data.iloc[:, 1:])

kmeans = KMeans(n_clusters=3)
kmeans.fit(data.iloc[:, 1:])

for no, cla in enumerate(kmeans.labels_):
    print("학생번호: {} : {}".format(no, cla))

print(kmeans.predict([[100,80,70,70,70]]))