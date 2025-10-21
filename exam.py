# 새로운 학생 예측 함수

def predict_gender(height, weight):

    global W, prediction

    inputs = np.array([height, weight, -1])

    prediction = step_func(np.dot(inputs, W))

    return '여성' if prediction == 1 else '남성'

# 예측 실행 및 결과 출력

new_height = 190

new_weight = 80

result = predict_gender(new_height, new_weight)

print(f"\n예측 결과: {new_height}cm, {new_weight}kg → {result}","prediction = ", prediction)