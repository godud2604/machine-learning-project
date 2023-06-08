## INFO
- 미션 : 날짜, 계절, 근무일 여부, 날짜, 온도, 체감 온도, 풍속 데이터를 활용하여 자전거 대여 수량 예측
- 문제 유형 : 회귀
- 평가 지표 : RMSLE
- 데이터 크기 : 1.1MB

## 학습 목표
- 자전거 대여 수요 예측을 위해 머신러닝 모델링 프로세스와 기본적인 회귀 모델 학습
- 그래프로 데이터를 시각화
- 회귀 모델 훈련/평가를 위한 학습
- 훈련된 모델을 이용하여 자전거 수요 예측

## 학습 순서
1. 탐색적 데이터 분석
    - 1-1. 데이터 둘러보기
    - 1-2. 피처 엔지니어링 (파생 피처 추가)
    - 1-3. 데이터 시각화
    - 1-4. 분석 정리 및 모델링 전략
2. 베이스라인 모델 (선형 회귀)
    - 1-1. 데이터 불러오기
    - 1-2. (기본적인)피처 엔지니어링
    - 1-3. 평가지표 계산 함수 작성
    - 1-4. 모델 훈련
    - 1-5. 성능 검증
    - 1-6. 제출
3. 성능 개선 I (릿지 회귀)
4. 성능 개선 II (라쏘 회귀)
5. 성능 개선 III (랜덤 포레스트 회귀)

## 학습 키워드
- 유형 및 평가지표 : 회귀, RMSLE
- 탐색적 데이터 분석 : 분포도, 막대 그래프, 박스플롯, 포인트플롯, 산점도, 히트맵
- 머신러닝 모델 : 선형 회귀, 릿지 회귀, 라쏘 회귀, 랜덤 포레스트 회귀
- 피처 엔지니어링 : 파생 피처 추가, 피처 제거
- 하이퍼파라미터 최적화 : 그리드서치