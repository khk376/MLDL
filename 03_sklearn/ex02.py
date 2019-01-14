# 다중 선형 회귀

from sklearn.linear_model import LinearRegression

# 공부한 시간, 학년
x= [[10, 3], [5, 2], [9, 3], [7, 3]]

#점수
y = [[100], [50], [90], [77]]

#학습기
model =LinearRegression()

model= model.fit(x,y)

result=model.predict([[9,2]])

print(result)

