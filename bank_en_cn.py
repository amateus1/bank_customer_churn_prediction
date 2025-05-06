import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

# Load model and encoders
model = tf.keras.models.load_model('model.h5')
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Language toggle
lang = st.selectbox("🌐 Select Language / 选择语言", ["English", "中文"])
is_cn = lang == "中文"

# UI labels
labels = {
    "title": "客户流失预测" if is_cn else "Customer Churn Prediction",
    "tab1": "单个预测" if is_cn else "Single Prediction",
    "tab2": "批量预测" if is_cn else "Batch Prediction",
    "upload": "上传CSV文件" if is_cn else "Upload CSV File",
    "download": "下载预测结果" if is_cn else "Download Results",
    "result": "预测结果" if is_cn else "Prediction Result",
    "prob": "客户流失概率" if is_cn else "Churn Probability",
    "flag": "流失?" if is_cn else "Churn?",
    "explain": "关键影响因素" if is_cn else "Key Risk Indicators"
}

st.title(labels["title"])
tab1, tab2 = st.tabs([labels["tab1"], labels["tab2"]])

# ─────────────── Tab 1: Single Prediction ─────────────── #
with tab1:
    geo = st.selectbox("地理位置 / Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("性别 / Gender", label_encoder_gender.classes_)
    age = st.slider("年龄 / Age", 18, 92)
    credit_score = st.number_input("信用评分 / Credit Score", 300, 850, 650)
    balance = st.number_input("账户余额 / Balance", 0.0, step=100.0)
    salary = st.number_input("预估薪资 / Estimated Salary", 0.0, step=100.0)
    tenure = st.slider("任职年限 / Tenure", 0, 10)
    products = st.slider("产品数量 / Number of Products", 1, 4)
    card = st.selectbox("是否持有信用卡 / Has Credit Card", [0, 1])
    active = st.selectbox("是否为活跃用户 / Is Active Member", [0, 1])

    df = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [products],
        'HasCrCard': [card],
        'IsActiveMember': [active],
        'EstimatedSalary': [salary]
    })

    geo_encoded = onehot_encoder_geo.transform([[geo]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    df = pd.concat([df.reset_index(drop=True), geo_df], axis=1)
    scaled = scaler.transform(df)

    proba = float(model.predict(scaled)[0][0])
    churn_flag = proba > 0.5

    st.subheader(labels["result"])
    st.metric(labels["prob"], f"{proba:.2f}")
    st.write(labels["flag"], "✅ 是" if churn_flag and is_cn else "❌ 否" if not churn_flag and is_cn else "Yes" if churn_flag else "No")

    fig, ax = plt.subplots(figsize=(6, 0.4))
    ax.barh([""], [proba], color="red" if churn_flag else "green")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([])
    st.pyplot(fig)

    st.subheader(labels["explain"])
    explain = []
    if credit_score < 600: explain.append("低信用评分 / Low Credit Score")
    if balance > 100000: explain.append("高账户余额 / High Balance")
    if active == 0: explain.append("非活跃用户 / Not Active")
    if not explain:
        explain.append("无明显风险 / No major red flags")
    for e in explain:
        st.markdown(f"- {e}")

# ─────────────── Tab 2: Batch Prediction ─────────────── #
with tab2:
    uploaded = st.file_uploader(labels["upload"], type="csv")
    if uploaded:
        try:
            data = pd.read_csv(uploaded)

            # Clean and normalize column names
            data.columns = data.columns.str.strip().str.replace(' ', '')

            # Drop any columns not used in model
            known_columns = [
                'Geography', 'Gender', 'Age', 'Tenure', 'CreditScore', 'Balance',
                'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
            ]
            data = data[[col for col in data.columns if col in known_columns or col.lower() in [k.lower() for k in known_columns]]]

            # Rename variants
            rename_map = {
                'NumofProducts': 'NumOfProducts',
            }
            data = data.rename(columns=rename_map)

            # Recheck required columns
            required_cols = known_columns
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Transform categorical
            data['Gender'] = label_encoder_gender.transform(data['Gender'])
            geo_enc = onehot_encoder_geo.transform(data[['Geography']]).toarray()
            geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
            geo_df = pd.DataFrame(geo_enc, columns=geo_cols)

            df_all = pd.concat([data.drop(['Geography'], axis=1).reset_index(drop=True), geo_df], axis=1)
            scaled_all = scaler.transform(df_all)
            pred_all = model.predict(scaled_all).flatten()

            result_df = data.copy()
            result_df[labels["prob"]] = pred_all
            result_df[labels["flag"]] = np.where(pred_all > 0.5, "是" if is_cn else "Yes", "否" if is_cn else "No")

            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(labels["download"], csv, file_name="predictions.csv", mime="text/csv")

        except Exception as e:
            st.error("⚠️ 数据格式错误或缺少必要列。" if is_cn else f"⚠️ Invalid format or missing columns: {e}")
            st.code(", ".join(known_columns))
