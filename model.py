# %%
import numpy as np
import pandas as pd
from datetime import datetime

train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')
submission = pd.read_csv('./data/sampleSubmission.csv')

# %%
"""
* 베이스라인 모델
- 선형 회귀 모델을 베이스라인으로 사용
- 베이스라인 모델에서 출발해 성능을 점차 향상시키는 방향으로 모델링할 것

* 피처 엔지니어링 전후의 데이터 합치기 및 나누기
- 훈련 데이터와 테스트 데이터를 합친다.
- 합친 데이터로 피처 엔지니어링 (타입 변경, 삭제, 추가)
- 다시 훈련 데이터와 테스트 데이터로 나누기.
"""

# %%
"""
이상치 제거
"""

# 훈련 데이터에서 weather가 4가 아닌 데이터만 추출 (폭우, 폭설이 내리는날 저녁 6시에 대여)
train = train[train['weather'] != 4]

# %%
"""
데이터 합치기
"""

all_data = pd.concat([train, test], ignore_index=True)
all_data

# %%
"""
파생 피처(변수) 추가
"""

# 날짜 피처 생성
all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])
# 연도 피처 생성
all_data['year'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0])
# 월 피처 생성
all_data['month'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1])
# 시 피처 생성
all_data['hour'] = all_data['datetime'].apply(lambda x: x.split()[1].split(':')[0])
# 요일 피처 생성
all_data['weekday'] = all_data['date'].apply(lambda dateString: datetime.strptime(dateString, "%Y-%m-%d").weekday())
# %%
