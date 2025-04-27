import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier
import shap
import matplotlib.pyplot as plt
import numpy as np
import joblib

# 加载模型
model = CatBoostClassifier().load_model("model/catboost_model.cbm")
explainer = joblib.load("model/explainer.shap")

# 定义特征列表
selected_features = ["Cortisol", "CRP", "Duration", "CysC", "C-P2", "BUN", "APTT", "RBG", "FT3", "ACR"]

# 设置页面标题
st.title('DR 预测模型及 SHAP 解释')

# 创建输入表单
st.sidebar.header('输入临床数据')
input_values = {}
for feature in selected_features:
    input_values[feature] = st.sidebar.number_input(feature)

# 构建输入数据
input_data = pd.DataFrame([input_values.values()], columns=selected_features)

# 进行预测
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# 显示预测结果
st.subheader('预测结果')
st.write(f'预测类别: {"DR" if prediction[0] == 1 else "DM"}')
st.write(f'预测概率: {prediction_proba[0][1]:.4f}')

# 计算 SHAP 值
shap_values = explainer.shap_values(input_data)

# 显示特征重要性解释
st.subheader('特征重要性解释')
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, input_data, plot_type="bar", max_display=10)
st.pyplot(plt.gcf())
plt.close()

# 显示单个样本解释
st.subheader('单个样本解释')
shap.force_plot(explainer.expected_value, shap_values[0], input_data.iloc[0], matplotlib=True, show=False)
st.pyplot(plt.gcf())
plt.close()
st.pyplot(bbox_inches='tight')