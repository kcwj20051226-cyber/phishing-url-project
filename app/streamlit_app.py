import streamlit as st
import joblib
import pandas as pd
import os
import sys

# 將專案根目錄加入系統路徑，以便順利匯入 src 中的模組
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import extract_features  # 直接使用你們原本寫好的特徵萃取函數！

# --- 1. 頁面基本設定 ---
st.set_page_config(
    page_title="智慧型釣魚網站偵測系統",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 側邊欄控制與導覽 ---
st.sidebar.title("🛡️ 導覽選單")
app_mode = st.sidebar.selectbox("選擇頁面功能", [
    "🔍 即時網址檢測 (URL Analysis)", 
    "📊 模型效能與圖表 (Model Performance)", 
    "📖 專案介紹與架構 (About Project)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ 設定")
model_choice = st.sidebar.selectbox("選擇預測模型", [
    "Random Forest (推薦/主力)", 
    "Decision Tree", 
    "Logistic Regression"
])

# 映射模型檔案路徑
model_map = {
    "Random Forest (推薦/主力)": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Logistic Regression": "models/logistic_regression.pkl"
}
selected_model_path = model_map[model_choice]

# 載入模型函式
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

model = load_model(selected_model_path)

# =========================================================
# 頁面一：即時網址檢測
# =========================================================
if app_mode == "🔍 即時網址檢測 (URL Analysis)":
    st.title("🛡️ 智慧型釣魚網站即時檢測系統")
    st.markdown("輸入可疑的網址（URL），系統將透過機器學習模型即時分析其特徵，並判斷是否為釣魚網站。")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        url_input = st.text_input("請輸入要檢測的網址：", placeholder="https://example.com/login-verify")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.button("開始分析 🚀", type="primary", use_container_width=True)

    if submit_btn:
        if not url_input.strip():
            st.warning("⚠️ 請先輸入有效的網址！")
        elif model is None:
            st.error(f"❌ 找不到模型檔案 `{selected_model_path}`，請先確認是否有執行過 `python src/train_models.py`！")
        else:
            with st.spinner("正在進行特徵萃取與模型預測..."):
                try:
                    # 1. 透過你們原生的 src/features.py 萃取特徵
                    features_dict = extract_features(url_input)
                    
                    # 2. 轉換成 DataFrame
                    input_df = pd.DataFrame([features_dict])
                    
                    # 3. 確保特徵欄位順序與模型訓練時完全一致
                    if hasattr(model, "feature_names_in_"):
                        # 如果模型有記錄特徵名稱，依照模型的順序重新排列欄位
                        input_df = input_df[model.feature_names_in_]
                    
                    # 4. 進行預測
                    pred = model.predict(input_df)[0]
                    proba = model.predict_proba(input_df)[0]
                    confidence = proba[pred] * 100
                    
                    st.markdown("---")
                    st.subheader("📋 檢測結果分析")
                    
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        if pred == 1:
                            st.error(f"### 🚨 警告：高度懷疑是釣魚網站 (Phishing)")
                            st.markdown(f"**模型信心分數**: `{confidence:.2f}%`")
                        else:
                            st.success(f"### ✅ 判定結果：此網址安全 (Legitimate)")
                            st.markdown(f"**模型信心分數**: `{confidence:.2f}%`")
                    
                    with res_col2:
                        st.info(f"**使用模型**: {model_choice}")
                        st.markdown(f"**網址長度**: {len(url_input)} 字元")

                    # 顯示萃取的特徵數值表
                    with st.expander("🔍 檢視萃取的網址特徵詳情 (Feature Values)"):
                        feat_df = pd.DataFrame(list(features_dict.items()), columns=["特徵名稱 (Feature)", "數值 (Value)"])
                        st.dataframe(feat_df, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ 預測過程發生錯誤: {e}")

# =========================================================
# 頁面二：模型效能與圖表
# =========================================================
elif app_mode == "📊 模型效能與圖表 (Model Performance)":
    st.title("📊 模型評估與視覺化分析")
    st.markdown("本頁面展示各機器學習模型在測試集上的效能表現、混淆矩陣以及關鍵特徵重要度。")
    
    comparison_csv = "outputs/tables/model_comparison.csv"
    if os.path.exists(comparison_csv):
        st.subheader("📋 模型效能比較總表")
        comp_df = pd.read_csv(comparison_csv)
        st.dataframe(comp_df, use_container_width=True)
    
    st.markdown("---")
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.subheader("📈 模型效能指標比較")
        if os.path.exists("outputs/figures/model_comparison.png"):
            st.image("outputs/figures/model_comparison.png", use_container_width=True)
            
        st.subheader("🎯 隨機森林混淆矩陣")
        if os.path.exists("outputs/figures/confusion_matrix_rf.png"):
            st.image("outputs/figures/confusion_matrix_rf.png", use_container_width=True)

    with fig_col2:
        st.subheader("⭐ 前 10 大關鍵特徵重要度")
        if os.path.exists("outputs/figures/feature_importance_rf.png"):
            st.image("outputs/figures/feature_importance_rf.png", use_container_width=True)
            
        st.subheader("🥧 總資料集類別分布")
        if os.path.exists("outputs/figures/label_distribution.png"):
            st.image("outputs/figures/label_distribution.png", use_container_width=True)

# =========================================================
# 頁面三：專案介紹與架構
# =========================================================
elif app_mode == "📖 專案介紹與架構 (About Project)":
    st.title("📖 專案背景與系統架構")
    
    st.markdown("""
    ### 🎯 專案動機
    隨著網路釣魚攻擊日益猖獗，使用者常因無法辨識精密的偽造網址而受害。本專案旨在透過**機器學習與網址特徵萃取技術**，建構一個能自動化辨識釣魚網站的智慧系統，並透過互動式網頁介面提供即時防護支援。
    
    ### ⚙️ 系統運作流程
    1. **資料前處理與特徵工程**：透過 `src/features.py` 統一萃取網址各項結構與關鍵字特徵。
    2. **多模型訓練與評估**：同時訓練 Decision Tree、Random Forest 與 Logistic Regression。
    3. **錯誤分析 (Error Analysis)**：深入檢討模型之 False Positive 與 False Negative 案例。
    4. **Web UI 展示平台**：使用 Streamlit 打造即時互動介面。
    """)
    
    st.markdown("---")
    st.markdown("### 👥 團隊成員")
    col1, col2, col3, col4, col5 = st.columns(5) # 宣告 5 個欄位
    col1.metric("113213021", "陳昱維")
    col2.metric("113213065", "謝承道")
    col3.metric("113213069", "蘇資晉")
    col4.metric("113213077", "温佳哲")
    col5.metric("113213076", "王志騰")