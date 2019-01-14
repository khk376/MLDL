# 선형 회귀 기본 예제
# 공부 시간으로 점수 예측해 보기
from sklearn.linear_model import LinearRegression

# 공부 시간
x= [[10], [5], [9], [7]]

# 점수
y = [[100], [50], [90], [77]]

model =LinearRegression()

model = model.fit(x,y)
result = model.predict([[7]])

print(result)