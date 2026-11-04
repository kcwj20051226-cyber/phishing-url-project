import os
import joblib
import pandas as pd
from src.features import extract_features  # 假設你們特徵萃取函數叫這個名字

def load_best_model(model_path="models/random_forest.pkl"):
    """載入訓練好的模型"""
    if not os.path.exists(model_path):
        # 如果找不到 random_forest 就退回決策樹
        model_path = "models/decision_tree.pkl"
    return joblib.load(model_path)

def predict_single_url(url, model_path="models/random_forest.pkl"):
    """
    輸入單一網址，回傳預測結果、機率與萃取出的特徵
    """
    # 1. 載入模型
    model = load_best_model(model_path)
    
    # 2. 萃取特徵 (將單一 URL 轉成 DataFrame 格式)
    # 註：請確保 features.py 裡有對應的單一 URL 萃取函數，或用你們原本寫的方式
    features_dict = extract_features(url) # 這裡視你們的 features.py 實作調整
    
    # 轉換成模型看得懂的 DataFrame 格式
    input_df = pd.DataFrame([features_dict])
    
    # 3. 進行預測與機率計算
    pred_label = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0] # [正常機率, 釣魚機率]
    
    # 4. 回傳標準格式
    return {
        "prediction_label": int(pred_label), # 0: 正常, 1: 釣魚
        "prediction_proba": float(pred_proba[pred_label]), # 該分類的信心分數
        "features_dict": features_dict
    }

if __name__ == "__main__":
    # 簡單測試 CLI 執行
    test_url = "http://example.com/login-verify-secure"
    result = predict_single_url(test_url)
    print(f"測試網址: {test_url}")
    print(f"預測結果: {'釣魚網站 (Phishing)' if result['prediction_label'] == 1 else '正常網站 (Legitimate)'}")
    print(f"信心分數: {result['prediction_proba']:.4f}")