import mglearn
import pandas as  pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans
import matplotlib

from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# raw data는 정제가 필요함(ex. 여백)
# read_csv로 해결
data = pd.read_csv("ex07_test.csv", sep = ",")
print(data)

# step 1: EDA
print("### step 1: EDA")
print(data.iloc[1, :1])
print(data.iloc[1, 1:2])
print(data.corr())
print(data.describe())
data.boxplot()
plt.show()

# step 2: 모델 학습
print("### step 2: 모델 학습")
kmeans = KMeans(n_clusters = 4)
kmeans.fit(data.iloc[:, :])
print("클러스터 레이블:", kmeans.labels_)

# step 3: 시각화
print("### step 3: 시각화")
mglearn.discrete_scatter(data.iloc[:, 0], data.iloc[:, 1], kmeans.labels_)
plt.legend(["cluster 0", "cluster 1", "cluster 2", "cluster 3"], loc = "best") # best는 보이는 범위 내에서 최적의 위치에 출력
plt.xlabel("ticket")
plt.ylabel("shopping")
plt.show()

# step 4: 예측
print("### step 4: 예측")
result = kmeans.predict([[2, 20]])
print(result)

# step 5: 최적 k값 찾기
print("### step 5: 최적 k값 찾기")
X = data.iloc[:, :]
y = kmeans.labels_
print(X, y)

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))