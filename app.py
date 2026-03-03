import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 設定網頁標題與風格
st.set_page_config(page_title="酒類分類預測系統", layout="wide")

# 加載資料集
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return wine, df

wine_data, df = load_data()

# --- Sidebar 區域 ---
st.sidebar.title("模型設定")

# 1. 模型選擇下拉選單
model_option = st.sidebar.selectbox(
    "請選擇預測模型：",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

st.sidebar.markdown("---")

# 2. 顯示酒類資料集資訊
st.sidebar.subheader("🍷 資料集資訊：Wine Dataset")
st.sidebar.info(f"""
**資料集描述：**
這是 sklearn 內建的酒類資料集，包含 3 種不同產地的葡萄酒化學分析結果。

- **樣本數：** {len(df)}
- **特徵數：** {len(wine_data.feature_names)}
- **類別數：** 3 (Class 0, 1, 2)
""")

# --- Main 區域 ---
st.title("🧪 酒類化學成分分析與分類預測")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 資料集前 5 筆內容")
    st.write(df.head())

with col2:
    st.subheader("📊 特徵統計值資訊")
    st.write(df.describe())

st.markdown("---")

# --- 預測邏輯 ---
st.subheader(f"🚀 模型預測：{model_option}")

if st.button("開始進行預測並執行評估"):
    # 準備資料
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 標準化 (對 KNN 與 羅吉斯迴歸特別重要)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 初始化模型
    if model_option == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_option == "羅吉斯迴歸":
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    elif model_option == "XGBoost":
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X_train, y_train) # XGBoost 不一定需要標準化
        y_pred = model.predict(X_test)
    elif model_option == "隨機森林":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # 計算準確度
    accuracy = accuracy_score(y_test, y_pred)
    
    # 顯示結果
    st.success(f"### 預測完成！")
    
    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("模型準確度 (Accuracy)", f"{accuracy:.2%}")
    
    st.write("#### 預測結果 vs 實際標籤 (前 10 筆測試集範例)")
    comparison_df = pd.DataFrame({
        '實際標籤': y_test.values[:10],
        '預測結果': y_pred[:10]
    })
    st.table(comparison_df)
    
    st.balloons()
else:
    st.write("請點擊上方按鈕以開始模型訓練與預測分析。")
