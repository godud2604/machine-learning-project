"""
[자전거 대여 수요 예측 경진대회 데이터의 피처들]
datetime : 기록 일시 (1시간 간격)
season : 계절 (1: 봄, 2: 여름, 3: 가을, 4: 겨울)
holiday : 공휴일 여부 (0: 공휴일 아님, 1: 공휴일)
workingday : 근무일 여부 (0: 근무일 아님, 1: 근무일 - 공휴일과 주말을 뺀 나머지 날)
weather : 날씨 (1: 맑음, 2: 옅은 안개, 3: 약간의 눈 또는 비와 천둥 번개, 4: 폭우와 천둥 번개 또는 눈과 짙은 안개)
temp : 실제 온도
atemp : 체감 온도
humidity : 상대 습도
windspeed : 풍속
casual : 등록되지 않은 사용자(비회원) 수
registered : 등록된 사용자(회원) 수

[예측해야 할 타깃값]
count : 자전거 대여 수량
"""

# %%
import numpy as np
import pandas as pd

from datetime import datetime
import calendar

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sampleSubmission.csv')

# %%
"""
테스트 데이터의 피처 casual과 registered가 제외된 상태.
=> 1. 훈련 데이터의 casual과 registered도 피처를 제거해야 한다.

ID 값(datetime)은 데이터를 구분하는 역할만 하므로 타깃값을 예측하는 데에 도움을 주지 않는다.
=> 2. 훈련 데이터의 datetime 피처 제거
"""
submission.head()

# %%
"""
datetime 피처의 데이터 타입은 object (판다스에서 object 타입은 문자열 타입이라고 보면 된다.)
datetime 세부적으로 분석하기 위해 구성요소별로 나눈다.
=> 연도, 월, 일, 시간, 분, 초, 피처 추가
(* 이처럼 기존 피처에서 파생된 피처를 "파생 피처" 혹은 "파생 변수" 라고 한다.)
"""
train.info()

# %%
train['date'] = train['datetime'].apply(lambda x: x.split()[0]) # 날짜 피처 생성

# 연도, 월, 일, 시, 분, 초 피처를 차례로 생성
train['year'] = train['datetime'].apply(lambda x: x.split()[0].split("-")[0]) 
train['month'] = train['datetime'].apply(lambda x: x.split()[0].split("-")[1]) 
train['day'] = train['datetime'].apply(lambda x: x.split()[0].split("-")[2]) 
train['hour'] = train['datetime'].apply(lambda x: x.split()[1].split(":")[0]) 
train['minute'] = train['datetime'].apply(lambda x: x.split()[1].split(":")[1]) 
train['second'] = train['datetime'].apply(lambda x: x.split()[1].split(":")[2]) 

# %%
"""
머신러닝 모델은 숫자만 인식하므로, 모델을 훈련할 때는 피처값을 문자로 바꾸면 안 된다.
"""

# 요일 피처 추가
train['weekday'] = train['date'].apply(
    lambda dateString:
    calendar.day_name[datetime.strptime(dateString, "%Y-%m-%d").weekday()])

"""
# season, weather
범주형 데이터인데 현재 1,2,3,4 라는 숫자로 표현되어 있어서 정확한 의미 파악 어렵다.
=> 시각화 시, 의미가 잘 드러나도록 문자열로 변경
"""

train['season'] = train['season'].map({ 1: 'Spring',
                                        2: 'Summer',
                                        3: 'Fall',
                                        4: 'Winter' })
train['weather'] = train['weather'].map({ 1: 'Clear',
                                          2: 'Mist, Few clouds',
                                          3: 'Light Snow, Rain, Thunderstorm',
                                          4: 'Heavy Rain, Thunderstorm, Snow, Fog'})

# %%
"""
데이터 시각화
=> 데이터 간 관계 파악
=> matplotlib, seaborn 라이브러리 활용

=> matplotlib : 파이썬으로 데이터를 시각화할 때 표준처럼 사용되는 라이브러리
=> seaborn : matplotlib에 고수준 인터페이스를 덧씌운 라이브러리
"""
mpl.rc('font', size=15)
sns.displot(train['count']) # 분포도 출력

"""
타깃값 count
=> 0 근처에 몰려 있다.
=> 분포가 왼쪽으로 많이 편향되어져 있다.
=> 회귀 모델이 좋은 성능을 내려면, 데이터가 정규분포를 따라야 하는데, 현재 타깃값 count는 정규분포를 따르지 않는다. (성능 안좋음)
"""

# %%
"""
로그 변환
=> 데이터 분포를 정규분포에 가깝이 만들기 위해 사용하는 방법
=> 후처리 : log(y)에서 실제 타깃값인 y를 복원하기 위해 지수변환을 해줘야 함
"""
sns.displot(np.log(train['count']))

# %%
"""
막대 그래프
- 각 범주형 데이터에 따라 평균 대여 수량이 어떻게 다른지 파악 (피처 중요도 확인)
- 아무 정보도 담고 있지 않은 피처는 모델 훈련에 사용하지 않는다

* day, minute, second 피처 제거
"""

# 1. m행 n열 Figure 준비
mpl.rc('font', size=14)
mpl.rc('axes', titlesize=15) # 각 축의 제목 크기 설정
figure, axes = plt.subplots(nrows=3, ncols=2) # 3행 2열 Figure 생성
plt.tight_layout() # 그래프 사이에 여백 확보
figure.set_size_inches(10, 9) # 전체 Figure 크기를 10x9인치로 설정

# 2. 각 축에 서브플롯 할당
sns.barplot(x='year', y='count', data=train, ax=axes[0, 0])
sns.barplot(x='month', y='count', data=train, ax=axes[0, 1])
sns.barplot(x='day', y='count', data=train, ax=axes[1, 0])
sns.barplot(x='hour', y='count', data=train, ax=axes[1, 1])
sns.barplot(x='minute', y='count', data=train, ax=axes[2, 0])
sns.barplot(x='second', y='count', data=train, ax=axes[2, 1])

# 3. 세부 설정
axes[0, 0].set(title="Rental amounts by year")
axes[0, 1].set(title="Rental amounts by month")
axes[1, 0].set(title="Rental amounts by day")
axes[1, 1].set(title="Rental amounts by hour")
axes[2, 0].set(title="Rental amounts by minute")
axes[2, 1].set(title="Rental amounts by second")

# 1행에 위치한 서브플롯들의 x축 라벨 90도 회전
axes[1, 0].tick_params(axis='x', labelrotation=90)
axes[1, 1].tick_params(axis='x', labelrotation=90)
# %%
"""
박스플롯
- 범주형 데이터에 따른 수치형 데이터 정보를 나타내는 그래프
- 막대 그래프보다 더 많은 정보를 제공하는 특징

* 공휴일이 아닐 때와 근무일일 때 이상치(outlier)가 많다.
"""

# 1. m행 n열 Figure 준비
figure, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
figure.set_size_inches(10, 10)

# 2. 서브플롯 할당
sns.boxplot(x='season', y='count', data=train, ax=axes[0, 0])
sns.boxplot(x='weather', y='count', data=train, ax=axes[0, 1])
sns.boxplot(x='holiday', y='count', data=train, ax=axes[1, 0])
sns.boxplot(x='workingday', y='count', data=train, ax=axes[1, 1])

# 3. 세부 설정
# 서브플롯에 제목 달기
axes[0, 0].set(title="Box Plot On Count Across Season")
axes[0, 1].set(title="Box Plot On Count Across Weather")
axes[1, 0].set(title="Box Plot On Count Across Holiday")
axes[1, 1].set(title="Box Plot On Count Across Working Day")

# x축 라벨 겹침 해결
axes[0, 1].tick_params(axis='x', labelrotation=10)
# %%

"""
포인트플롯
- 범주형 데이터에 따른 수치형 데이터의 평균과 신뢰구간을 점과 선으로 표시
- 막대 그래프와 동일한 정보를 제공하지만, 한 화면에 여러 그래프를 그려 서로 비교해보기에 더 적합하다.

* weather == 4인 데이터 제거 (이상치)
"""

# 1. m행 n열 Figure 준비
mpl.rc('font', size=11)
figure, axes = plt.subplots(nrows=5) # 5행 1열
figure.set_size_inches(12, 18)

# 2. 서브플롯 할당
# 근무일, 공휴일, 요일, 계절, 날씨에 따른 시간대별 평균 대여 수량 포인트플롯
# hue 파라미터에 비교하고 싶은 피처 전달
sns.pointplot(x='hour', y='count', data=train, hue='workingday', ax=axes[0])
sns.pointplot(x='hour', y='count', data=train, hue='holiday', ax=axes[1])
sns.pointplot(x='hour', y='count', data=train, hue='weekday', ax=axes[2])
sns.pointplot(x='hour', y='count', data=train, hue='season', ax=axes[3])
sns.pointplot(x='hour', y='count', data=train, hue='weather', ax=axes[4])
# %%
"""
회귀선을 포함한 산점도 그래프
- 데이터 간 상관관계를 파악하는 데 사용
- 회귀선 기울기로 대략적인 추세를 파악할 수 있다.

* windspeed 피처 제거
- 결측치가 많다. (풍속이 0인 데이터 다수 존재)
"""

# 1. m행 n열 Figure 준비
mpl.rc('font', size=15)
figure, axes = plt.subplots(nrows=2, ncols=2)
plt.tight_layout()
figure.set_size_inches(7, 6)

# 2. 서브플롯 할당
# 온도, 체감 온도, 풍속, 습도 별 대여 수량 산점도 그래프
sns.regplot(x='temp', y='count', data=train, ax=axes[0, 0],
            scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})
sns.regplot(x='atemp', y='count', data=train, ax=axes[0, 1],
            scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})
sns.regplot(x='windspeed', y='count', data=train, ax=axes[1, 0],
            scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})
sns.regplot(x='humidity', y='count', data=train, ax=axes[1, 1],
            scatter_kws={'alpha': 0.2}, line_kws={'color': 'blue'})



# %%
"""
히트맵
- 수치형 데이터끼리 어떤 상관관계가 있는지 확인 가능
- 피처들의 수많은 조합 사이의 관계를 한눈에 파악 가능.

* windspeed와 count의 상관관계는 0.1
=> 상관관계가 매우 약하다. 즉, 수량 예측에 큰 도움이 안 됨
=> windspeed 피처 제거
"""

# 피처 간 상관관계 매트릭스
corrMat = train[['temp', 'atemp', 'humidity', 'windspeed', 'count']].corr()
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
sns.heatmap(corrMat, annot=True) 
ax.set(title="Heatmap of Numerical Data")