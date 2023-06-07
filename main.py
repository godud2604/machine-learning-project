"""
[자전거 대여 수요 예측 경진대회 데이터의 피처들]
datetime : 기록 일시 (1시간 간격)
season : 계절 (1: 봄, 2: 여름, 3: 가을, 4: 겨울)
holiday : 공휴일 여부 (0: 공휴일 아님, 1: 공휴일)
workingday : 근무일 여부 (0: 근무일 아님, 1: 근무일)
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
"""
sns.displot(np.log(train['count']))
# %%
