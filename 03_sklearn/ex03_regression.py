from sklearn.datasets import load_boston
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 데이터 활용
boston = load_boston()
dfx = pd.DataFrame(boston.data, columns=boston.feature_names)
dfy = pd.DataFrame(boston.target, columns=['MEDV']) # 보스턴 주택 가격
df = pd.concat([dfx, dfy], axis=1)


#print(df.head())
print(df)
'''
1. 데이터 셋에 대한 사전 조사
2. 누락 데이터 여부
3. 각 데이터가 연속적인지, 실수인지, 범주형인지 등등
4. 데이터간의 상관관계
5. 이상치 등도 확인 등 ...
'''
