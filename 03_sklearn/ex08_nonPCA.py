# 유방암 진단 데이터 셋을 사용해 환자가 음성/양성인지 확인
# pca 미적용된 test예제
# 목적 : 유방암 진단 데이터셋 시각화해보기

import mglearn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

# 한글 보이게 하는 설정
'''
font_location = 'C:/Windows/fonts/malgun.ttf'
font_name = font_manager.FontProperties(font = font_location).get_name()
matplotlib.rc('font', family = font_name)
'''

# 타겟 0 => 악성 / 타겟1 => 양성
cancer = load_breast_cancer()
print(cancer)

# 변수 30개, 그래프 띄울 때, 15 x 2로 객체 생성

fig, axes = plt.subplots(15, 2, figsize = (10, 20))

# 악성 데이터
maligant = cancer.data[cancer.target == 0]

# 양성 데이터
benign = cancer.data[cancer.target == 1]


# 다차원 배열을 1차원 배열로 재구성
ax = axes.ravel()


# histogram 그리기
# bin : 막대그래프에서 x축의 폭 또는 interval, cell, 구간

for i in range(30):
    _, bins = np.histogram(cancer.data[:, i], bins = 50)   # _, 은 버려지는 데이터??
    ax[i].hist(maligant[:, i], bins = bins, color = 'r', alpha = 0.5)
    ax[i].hist(benign[:, i], bins = bins, color = 'b', alpha = 0.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())

ax[0].set_xlabel('feature_size')
ax[0].set_ylabel('frequency')
ax[0].legend(['maligant', 'benign'], loc = 'best')

fig.tight_layout()
plt.show()