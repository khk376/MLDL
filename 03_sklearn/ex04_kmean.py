# K-means 알고리즘을 육안으로 확인 가능한 그림
# 클러스터들의 중심점을 가장 가까운 거리의 중심점이 될 수 있도록 반복하여 이동하는 모습 확인 가능
# 반복하여 이동하는 모습 확인 가능
# pip install mglearn

''' k-Nearest Neighbor
k-means 알고리즘
1. 가장 간단하고 널리 알려진 군집(클러스터링) 알고리즘
2. 클러스터 중심을 찾는 알고리즘
3. 데이터 포인트를 가장 가까운 클래스터 중심에 할당하고 
   거리 평균으로 클러스터 중심을 다시 지정


'''


import mglearn
import matplotlib.pyplot as plt

#kmeans 알고리즘 동작을 설명하는 그림 그리기
mglearn.plots.plot_kmeans_algorithm()
plt.show()

# 경계선 그리기
mglearn.plots.plot_kmeans_boundaries()
plt.show()

