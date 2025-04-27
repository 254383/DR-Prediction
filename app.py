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

# 页面配置
st.set_page_config(page_title="DR Prediction", layout="wide")
st.title("糖尿病视网膜病变风险评估系统")

# 输入表单
with st.sidebar:
    st.header("输入临床指标")
    inputs = {}
    features = ["Cortisol", "CRP", "Duration", "CysC", "C-P2", "BUN", "APTT", "RBG", "FT3", "ACR"]
    for feat in features:
        inputs[feat] = st.number_input(feat, value=0.0)

# 预测与解释
if st.button("开始评估"):
    input_df = pd.DataFrame([inputs])

    # 预测概率
    prob = model.predict_proba(input_df)[0][1]
    st.success(f"DR风险概率: {prob * 100:.1f}%")

    # SHAP解释
    st.subheader("特征影响分析")
    shap_value = explainer.shap_values(input_df)

    # 绘制Force Plot
    plt.figure()
    shap.force_plot(
        explainer.expected_value,
        shap_value[0],
        input_df.iloc[0],
        matplotlib=True,
        show=False
    )
    st.pyplot(plt.gcf())

    # 特征重要性表格
    st.subheader("特征贡献度排名")
    contribution_df = pd.DataFrame({
        "Feature": features,
        "SHAP Value": shap_value[0]
    }).sort_values("SHAP Value", ascending=False)
    st.dataframe(contribution_df)
    import os

    # 动态获取当前脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model", "catboost_model.cbm")

    # 加载模型
    model = CatBoostClassifier().load_model(model_path)
