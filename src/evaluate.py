import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from features import extract_features
from rule_based import rule_based_predict

print("正在進行最後的盲測評估 (Test Set Evaluation)...")

# 1. 載入測試集資料
test_df = pd.read_csv("data/processed/test.csv")
test_feat_df = pd.read_csv("data/processed/test_features.csv")

y_true = test_df["label"].values

# ==========================================
# 2. 評估方法一：規則法 (Rule-based Baseline)
# ==========================================
y_pred_rule = []
for url in test_df["url"]:
    _, pred, _ = rule_based_predict(url)
    y_pred_rule.append(pred)

rule_acc = accuracy_score(y_true, y_pred_rule)
rule_prec = precision_score(y_true, y_pred_rule, zero_division=0)
rule_rec = recall_score(y_true, y_pred_rule, zero_division=0)
rule_f1 = f1_score(y_true, y_pred_rule, zero_division=0)
rule_cm = confusion_matrix(y_true, y_pred_rule)

# ==========================================
# 3. 評估方法二：決策樹模型 (Decision Tree)
# ==========================================
model_path = "models/decision_tree.joblib"
if not os.path.exists(model_path):
    print("找不到訓練好的決策樹模型，請先執行第 8 步的 train_decision_tree.py！")
    exit()

dt_model = joblib.load(model_path)
X_test = test_feat_df.drop(columns=["url", "label"])
y_pred_dt = dt_model.predict(X_test)

dt_acc = accuracy_score(y_true, y_pred_dt)
dt_prec = precision_score(y_true, y_pred_dt, zero_division=0)
dt_rec = recall_score(y_true, y_pred_dt, zero_division=0)
dt_f1 = f1_score(y_true, y_pred_dt, zero_division=0)
dt_cm = confusion_matrix(y_true, y_pred_dt)

# ==========================================
# 4. 整理成計畫書要求的結果表格
# ==========================================
results_data = {
    "方法 (Method)": ["Rule-based (Baseline)", "Decision Tree (Model)"],
    "Precision": [round(rule_prec, 4), round(dt_prec, 4)],
    "Recall": [round(rule_rec, 4), round(dt_rec, 4)],
    "F1-score": [round(rule_f1, 4), round(dt_f1, 4)],
    "Accuracy": [round(rule_acc, 4), round(dt_acc, 4)],
}

results_df = pd.DataFrame(results_data)

print("\n" + "=" * 50)
print("📊 最終測試集評估比較表 (Test Set Performance)")
print("=" * 50)
print(results_df.to_string(index=False))
print("=" * 50)

# 5. 印出混淆矩陣 (Confusion Matrix) 供參考
print("\n📌 混淆矩陣說明 ([ [TN, FP], [FN, TP] ])：")
print("【規則法 Rule-based 混淆矩陣】：")
print(rule_cm)
print("\n【決策樹 Decision Tree 混淆矩陣】：")
print(dt_cm)

# 6. 自動輸出成 CSV 報表，方便直接貼進期中報告書或簡報
os.makedirs("reports", exist_ok=True)
report_output = "reports/evaluation_comparison.csv"
results_df.to_csv(report_output, index=False)
print(f"\n💾 評估比較表已成功儲存至：{report_output}")