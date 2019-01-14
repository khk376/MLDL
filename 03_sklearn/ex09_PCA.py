
import mglearn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd

cancer = load_breast_cancer()

# 1단계 pca 적용전에 각 특성의 분산이 1이 되도록 데이터의 스케일 조정
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(cancer.data)
X_Scaled=scaler.transform(cancer.data)

# 2단계 : PCA 적용, 두개의 주성분만 유지하는 데이터로 변환(차원 축소)
from sklearn.decomposition import PCA
pca= PCA(n_components=2)
pca.fit(X_Scaled)

X_pca=pca.transform(X_Scaled)

print(X_pca.shape)

# 3단계 : 시각화 (산점도. 악성:0, 양성:1) 
import mglearn
import matplotlib.pyplot as plt 

mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(['maligant', 'benign'], loc="best")
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.show()

# 4단계 : cacner.target 사용하지 않고 주성분 두가지로 군집화
from sklearn.cluster import KMeans
kmeans= KMeans(n_clusters=2)
kmeans.fit(X_Scaled)
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], kmeans.labels_, markers='^') # KMeans.labels_ 에서 언더바 쓰면 각각 특성이 반영됨
plt.legend(['maligant', 'benign'], loc="best")
plt.xlabel('1st component')
plt.ylabel('2nd component')
plt.show()

