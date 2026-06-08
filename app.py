# Streamlit 앱 파일 생성

import streamlit as st  # Streamlit 사용
import pandas as pd  # 데이터프레임 사용
import joblib  # pkl 파일 사용
import matplotlib.pyplot as plt  # 그래프 사용

linear_model = joblib.load("linear_model.pkl")  # Linear 로드

poly_model = joblib.load("poly_model.pkl")  # Poly 로드

ridge_model = joblib.load("ridge_model.pkl")  # Ridge 로드

model_info = joblib.load("model_info.pkl")  # 정보 로드

features = model_info["features"]  # 독립변수 목록

performance_df = model_info["performance_df"]  # 성능표

feature_min = model_info["feature_min"]  # 최솟값

feature_max = model_info["feature_max"]  # 최댓값

feature_mean = model_info["feature_mean"]  # 평균값

models = {  # 모델 모음
    "Linear": linear_model,  # Linear
    "Poly": poly_model,  # Poly
    "Ridge": ridge_model  # Ridge
}  # 모델 모음 끝

st.title("WHO 기대수명 예측 웹 서비스")  # 제목 출력

st.write("다중 특성 회귀 모델로 기대수명을 예측합니다.")  # 설명 출력

st.write("---")  # 구분선

st.header("1. 모델 성능 비교")  # 성능 제목

st.write("Train/Test R2, MSE, Complexity 비교표입니다.")  # 표 설명

st.dataframe(performance_df)  # 성능표 출력

fig, ax = plt.subplots(figsize=(7, 4))  # 그래프 영역

ax.bar(performance_df["Model"], performance_df["Test R2"])  # Test R2 그래프

ax.set_title("Test R2 Score Comparison")  # 제목

ax.set_xlabel("Model")  # x축

ax.set_ylabel("Test R2 Score")  # y축

ax.grid(axis="y")  # 격자

st.pyplot(fig)  # 그래프 출력

st.write("---")  # 구분선

st.header("2. 기대수명 실시간 예측")  # 예측 제목

st.sidebar.header("입력값 조절")  # 사이드바 제목

input_data = {}  # 입력값 저장

for feature in features:  # 특성 반복
    min_value = float(feature_min[feature])  # 최솟값

    max_value = float(feature_max[feature])  # 최댓값

    mean_value = float(feature_mean[feature])  # 기본값

    step_value = float((max_value - min_value) / 100)  # 이동 간격

    input_data[feature] = st.sidebar.slider(  # 슬라이더 생성
        feature,  # 이름
        min_value,  # 최솟값
        max_value,  # 최댓값
        mean_value,  # 기본값
        step_value  # 간격
    )  # 슬라이더 끝

selected_model_name = st.selectbox(  # 모델 선택
    "모델 선택",  # 안내 문구
    ["Linear", "Poly", "Ridge"]  # 선택 목록
)  # 선택 끝

selected_model = models[selected_model_name]  # 선택 모델

input_df = pd.DataFrame([input_data])  # 입력값 표 생성

prediction = selected_model.predict(input_df)[0]  # 기대수명 예측

st.header("예측 결과")  # 결과 제목

st.header(str(round(prediction, 2)) + " 세")  # 큰 글씨 출력
