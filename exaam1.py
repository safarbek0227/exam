mport numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

# 훈련 데이터 (키, 체중, 편향)

# 로지스틱 회귀 모델은 Bias(편향)를 자동으로 추가하므로, 입력 데이터 X에서 Bias(-1)를 제외합니다.

X = np.array([

    [170, 80],

    [175, 76],

    [180, 70],

    [160, 55],

    [163, 43],

    [165, 48]

])

# 레이블 (0: 남성, 1: 여성)

Y = np.array([0, 0, 0, 1, 1, 1])

# -------------------------------------------------------------

# 1. 로지스틱 회귀 모델 구축 및 학습

# -------------------------------------------------------------

# solver='liblinear'는 작은 데이터셋에 적합하며, L2 규제를 기본으로 합니다.

# C=100은 규제 강도를 낮춰(큰 값), 모델이 데이터에 더 잘 맞도록 합니다.

model = LogisticRegression(solver='liblinear', C=100, random_state=42)

# 모델 학습

model.fit(X, Y)

print("로지스틱 회귀 모델 학습 완료.")

# -------------------------------------------------------------

# 2. 결과 예측 및 평가

# -------------------------------------------------------------

# 학습된 데이터를 다시 예측에 사용

Y_pred = model.predict(X)

accuracy = accuracy_score(Y, Y_pred)

print("\n----------------- 학습 결과 -----------------")

print("실제 레이블 (Y):", Y)

print("모델 예측값:", Y_pred)

print(f"정확도 (Accuracy): {accuracy * 100:.2f}%")

# -------------------------------------------------------------

# 3. 결정 경계 시각화

# -------------------------------------------------------------

# 모델의 가중치와 절편 추출 (로지스틱 회귀 결정 경계: W1*x1 + W2*x2 + b = 0)

W1, W2 = model.coef_[0]

b = model.intercept_[0]

# 축 범위 설정

x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5

y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

# 그래프 생성

plt.figure(figsize=(10, 8))

# 데이터 포인트 산점도

plt.scatter(X[Y == 0, 0], X[Y == 0, 1], color='blue', marker='o', label='Class 0 (남성)')

plt.scatter(X[Y == 1, 0], X[Y == 1, 1], color='red', marker='x', label='Class 1 (여성)')

# 결정 경계선 그리기 (y = (-W1/W2) * x - (b/W2))

x_vals = np.array([x_min, x_max])

# W2가 0에 가까울 때 예외 처리 필요하지만, 로지스틱 회귀에서는 흔치 않음

if abs(W2) > 1e-8:

    y_vals = (-W1 * x_vals - b) / W2

    plt.plot(x_vals, y_vals, color='green', linestyle='-', label='Logistic Regression Boundary')

else:

    # 거의 수직인 경우

    x_val = -b / W1

    plt.axvline(x=x_val, color='green', linestyle='-', label='Logistic Regression Boundary')

# 그래프 최종 설정

plt.xlim(x_min, x_max)

plt.ylim(y_min, y_max)

plt.title('Logistic Regression Decision Boundary (Height vs. Weight)')

plt.xlabel('Height (cm)')

plt.ylabel('Weight (kg)')

plt.grid(True)

plt.legend()

plt.show()