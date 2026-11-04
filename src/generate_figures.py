import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 使用預設的標準英文字型，徹底避免 Windows 亂碼或方框問題
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")

print("🎨 開始使用英文標籤繪製期中專案視覺化圖表...")

# 確保輸出目錄存在
os.makedirs("outputs/figures", exist_ok=True)

# 1. 載入資料與模型
train_path = "data/processed/train_features.csv"
test_path = "data/processed/test_features.csv"

if not os.path.exists(test_path):
    print("❌ 找不到測試集特徵檔案，請先確認 data/processed/ 底下是否有對應檔案！")
    exit()

test_df = pd.read_csv(test_path)
X_test = test_df.drop(columns=["url", "label"])
y_test = test_df["label"]

# 載入我們的主力模型 (Random Forest)
rf_model_path = "models/random_forest.pkl"
if os.path.exists(rf_model_path):
    rf_model = joblib.load(rf_model_path)
else:
    print("❌ 找不到隨機森林模型，請先執行 src/train_models.py！")
    exit()

# ---------------------------------------------------------
# 圖表 1：模型比較長條圖 (Model Comparison Bar Chart)
# ---------------------------------------------------------
print("  正在繪製：Model Comparison Bar Chart (model_comparison.png) ...")
comparison_csv = "outputs/tables/model_comparison.csv"
if os.path.exists(comparison_csv):
    df_comp = pd.read_csv(comparison_csv)
    
    df_melted = df_comp.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score"],
                             var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="Blues")
    plt.title("Model Performance Comparison", fontsize=16, fontweight="bold", pad=15)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig("outputs/figures/model_comparison.png", dpi=300)
    plt.close()

# ---------------------------------------------------------
# 圖表 2：混淆矩陣熱力圖 (Confusion Matrix Heatmap - Random Forest)
# ---------------------------------------------------------
print("  正在繪製：Confusion Matrix (confusion_matrix_rf.png) ...")
y_pred_rf = rf_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"],
            annot_kws={"size": 14, "weight": "bold"})
plt.title("Random Forest - Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/figures/confusion_matrix_rf.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 圖表 3：特徵重要度排行圖 (Feature Importance Bar Chart)
# ---------------------------------------------------------
print("  正在繪製：Feature Importance (feature_importance_rf.png) ...")
importances = rf_model.feature_importances_
features = X_test.columns

feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False).head(10) # 取前 10 名

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_df, x="Importance", y="Feature", palette="viridis")
plt.title("Random Forest - Top 10 Feature Importances", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature Name", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/figures/feature_importance_rf.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# 圖表 4：資料集標籤分布圖 (Label Distribution)
# ---------------------------------------------------------
print("  正在繪製：Label Distribution (label_distribution.png) ...")
if os.path.exists(train_path):
    train_df = pd.read_csv(train_path)
    full_labels = pd.concat([train_df["label"], test_df["label"]])

    label_counts = full_labels.value_counts().rename({0: "Legitimate", 1: "Phishing"})

    plt.figure(figsize=(6, 6))
    plt.pie(label_counts, labels=label_counts.index, autopct="%1.1f%%", startangle=140,
            colors=["#66b3ff", "#ff9999"], explode=(0.05, 0), shadow=True)
    plt.title("Dataset Class Distribution (Benign vs Phishing)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig("outputs/figures/label_distribution.png", dpi=300)
    plt.close()

print("\n🎉 所有英文視覺化圖表已成功產出並存至 outputs/figures/ 資料夾！")