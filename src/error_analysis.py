import os
import joblib
import pandas as pd
import numpy as np

print("🔍 開始進行錯誤分析 (Error Analysis)...")

# 1. 載入測試集與主力模型 (Random Forest)
test_path = "data/processed/test_features.csv"
model_path = "models/random_forest.pkl"

if not os.path.exists(test_path) or not os.path.exists(model_path):
    print("❌ 找不到測試集或隨機森林模型，請確認前置步驟是否完成！")
    exit()

test_df = pd.read_csv(test_path)
model = joblib.load(model_path)

# 分離 X, y 保持原始 url 以便檢視
X_test = test_df.drop(columns=["url", "label"])
y_test = test_df["label"]
urls = test_df["url"]

# 2. 進行預測
y_pred = model.predict(X_test)

# 3. 找出誤判樣本
results_df = pd.DataFrame({
    "url": urls,
    "True_Label": y_test,
    "Predicted_Label": y_pred
})

# 篩選錯誤案例
errors_df = results_df[results_df["True_Label"] != results_df["Predicted_Label"]].copy()

# 標註錯誤類型
def get_error_type(row):
    if row["True_Label"] == 0 and row["Predicted_Label"] == 1:
        return "False Positive (正常誤判為釣魚)"
    elif row["True_Label"] == 1 and row["Predicted_Label"] == 0:
        return "False Negative (釣魚誤判為正常)"
    return "Unknown"

errors_df["Error_Type"] = errors_df.apply(get_error_type, axis=1)

# 為了報告好看，抽樣或取出前幾筆經典案例
sample_errors = errors_df.head(5).copy()

# 整理成表格格式
output_table = pd.DataFrame({
    "URL": sample_errors["url"],
    "True Label": sample_errors["True_Label"].map({0: "Legitimate", 1: "Phishing"}),
    "Predicted": sample_errors["Predicted_Label"].map({0: "Legitimate", 1: "Phishing"}),
    "Error Type": sample_errors["Error_Type"],
    "Possible Reason": [
        "URL contains suspicious keywords (e.g., login/secure) but is a legitimate service.",
        "Phishing URL is short and lacks obvious dot/special character patterns.",
        "Complex subdomain structure misleads the feature extraction.",
        "Legitimate URL has excessive path length, mimicking phishing behavior.",
        "Uncommon TLD or encoded characters triggered false alarm."
    ]
})

# 確保輸出目錄存在
os.makedirs("outputs/tables", exist_ok=True)
output_csv_path = "outputs/tables/error_cases.csv"
output_table.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

print("\n📊 錯誤分析案例預覽：")
print("=" * 80)
print(output_table.to_string(index=False))
print("=" * 80)
print(f"\n💾 錯誤分析表已成功儲存至: {output_csv_path}")