import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os
import joblib
# 获取当前文件的路径
current_dir = os.path.dirname(__file__)

# 模型和标准化器的文件路径
model_path = os.path.join(current_dir, 'rf_model.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# 加载模型和新的标准化器
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 所有特征名称
all_features = ['Height', 'Weight', 'SAA', 'CRP', 'N_number', 'N_percentage', 'M_number','M_percentage',
                'Bas_number','Bas_percentage','Eos_number','Eos_percentage','MCV','MCH','MCHC',
                'MPV','L_number','L_percentage','WBC','RDW','Hct','RBC','PDW','PLT','HB',
                'Fever','Typer of Fever','Peak tempreture','Cough','Gasp','shiluoyin','xiaomingyin',
                'Age','Sex']

# 最终模型训练使用的特征名称
selected_features = ['Cough', 'SAA', 'CRP', 'Gasp', 'Height', 'RDW', 'Age', 'M_number', 'N_number', 'WBC']

# 初始化页面
st.title('肺炎支原体混合细菌感染诊断模型')
st.text("请输入此次病程中最早的血常规指标和入院时的症状，若出现CRP<0.8之类的描述，请填入CRP=0.4，若出现SAA>320之类的描述，请填入SAA=320")
# 创建用户输入控件
Cough = st.number_input('咳嗽时间（天）', value=0.0)
SAA = st.number_input('血清淀粉样蛋白', value=0.0)
CRP = st.number_input('C反应蛋白', value=0.0)
Gasp = st.number_input('喘息时长（天）', value=0.0)
Height = st.number_input('身高（cm）', value=0.0)
RDW = st.number_input('红细胞分布宽度', value=0.0)
Age = st.number_input('年龄（月）', value=0.0)
M_number = st.number_input('单核细胞计数', value=0.0)
N_number = st.number_input('中性粒细胞计数', value=0.0)
WBC = st.number_input('白细胞计数', value=0.0)
# 将输入特征存储在字典中，并将未使用的特征填充为0
input_features = {
    'Cough': Cough,
    'SAA': SAA,
    'CRP': CRP,
    'Gasp': Gasp,
    'Height': Height,
    'RDW': RDW,
    'Age ': Age,
    'M_number': M_number,
    'N_number': N_number,
    'WBC': WBC,
}

# 确保所有特征都存在，未使用的特征填充为0
for feature in all_features:
    if feature not in input_features:
        input_features[feature] = 0.0

# 构建包含所有特征的数组
features = np.array([[input_features[feature] for feature in all_features]])

# 标准化特征值
scaled_features = scaler.transform(features)

# 仅选择模型所需的特征进行预测
scaled_features_selected = scaled_features[:, [all_features.index(feature) for feature in selected_features]]

# 预测
predicted_probs = model.predict_proba(scaled_features_selected)
aki_probability = predicted_probs[0][1]

if st.button('Diagnose'):
    st.markdown(f"<h3>患者属于肺炎支原体混合细菌感染的概率为 <span style='color:red;'>{aki_probability * 100:.2f}%</span></h3>", unsafe_allow_html=True)
    st.markdown("参考截断值为12%，在此截断值下灵敏度为0.981，特异性为0.839", unsafe_allow_html=True)
