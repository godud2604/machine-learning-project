# %%
import numpy as np
import pandas as pd

from datetime import datetime
from sklearn.linear_model import LinearRegression

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
"""
필요 없는 피처 제거
"""

drop_features = ['casual', 'registered', 'datetime', 'date', 'month', 'windspeed']

all_data = all_data.drop(drop_features, axis=1)

# %%
"""
데이터 나누기
"""
# 훈련 데이터와 테스트 데이터 나누기
X_train = all_data[~pd.isnull(all_data['count'])]
X_test = all_data[pd.isnull(all_data['count'])]

# 타깃값 count 제거
X_train = X_train.drop(['count'], axis=1)
X_test = X_test.drop(['count'], axis=1)

# 타깃값
y = train['count']

# %%
"""
평가지표 계산 함수 작성
- 훈련이 제대로 이루어졌는지 확인하기위한 평가 지표 (훈련에 앞서 평가지표인 RMSLE를 계산하는 함수 생성)

* convertExp : 입력 데이터를 지수변환할지를 정하는 파라미터
"""

def rmsle(y_true, y_pred, convertExp=True):
    # 지수변환
    if convertExp:
        y_true = np.exp(y_true)
        y_pred = np.exp(y_pred)

    # 로그변환 후 결측값을 0으로 변환
    log_true = np.nan_to_num(np.log(y_true+1))
    log_pred = np.nan_to_num(np.log(y_pred+1))

    # RMSLE 계산
    output = np.sqrt(np.mean((log_true - log_pred) ** 2))
    
    return output

# %%
"""
모델 훈련
- 모델을 생성한 뒤, 훈련
- 선형 회귀 모델 훈련

- 훈련 : 피처(독립변수)와 타깃값(종속변수)이 주어졌을 때 최적의 가중치(회귀계수)를 찾는 과정
- 예측 : 최적의 가중치를 아는 상태(훈련된 모델)에서 새로운 독립변수(데이터)가 주어졌을 때 타깃값을 추정하는 과정
"""

linear_reg_model = LinearRegression()

log_y = np.log(y) # 타깃값 로그변환
linear_reg_model.fit(X_train, log_y)

# %%
"""
모델 성능 검증
- 훈련을 마쳤으니 예측을 해본 후 RMSLE 값 확인
"""

preds = linear_reg_model.predict(X_train)

print(f'선형 회귀의 RMSLE 값 : {rmsle(log_y, preds, True):.4f}')

# %%
"""
예측 및 결과 제출
1. 테스트 데이터로 예측한 결과를 이용해야 한다.
2. 예측한 값에 지수변환을 해줘야 한다.
"""

linearreg_preds = linear_reg_model.predic(X_test)

submission['count'] = np.exp(linearreg_preds)
submission.to_csv('submission.csv', index=False) # DataFrame을 csv파일로 저장
