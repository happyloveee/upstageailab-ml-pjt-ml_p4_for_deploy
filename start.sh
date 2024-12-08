#!/bin/bash

# MLflow 서버 백그라운드로 시작
mlflow server \
    --host 0.0.0.0 \
    --port 5050 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns &

# Streamlit 앱 시작
streamlit run app.py