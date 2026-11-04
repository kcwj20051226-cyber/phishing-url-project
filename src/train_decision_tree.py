import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. 讀取前面產出的特徵表
print("正在載入訓練集與驗證集特徵表...")
train_df = pd.read_csv("data/processed/train_features.csv")
valid_df = pd.read_csv("data/processed/valid_features.csv")

# 2. 分離特徵 (X) 與標籤 (y) (排除 url 欄位)
X_train = train_df.drop(columns=["url", "label"])
y_train = train_df["label"]

X_valid = valid_df.drop(columns=["url", "label"])
y_valid = valid_df["label"]

# 3. 建立並訓練決策樹模型 (設定 max_depth=10 避免過擬合，符合計畫書要求)
print("\n🌳 正在訓練決策樹模型 (Decision Tree)...")
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)

# 4. 在驗證集 (Validation Set) 上評估效果
y_valid_pred = dt_model.predict(X_valid)
valid_acc = accuracy_score(y_valid, y_valid_pred)

print("\n--- 驗證集 (Validation Set) 評估結果 ---")
print(f"正確率 (Accuracy): {valid_acc:.4f}")
print("\n詳細分類報告：")
print(classification_report(y_valid, y_valid_pred, target_names=["正常 (0)", "釣魚 (1)"]))

# 5. 輸出特徵重要度 (Feature Importance)，這在期中簡報超好講！
print("\n📊 特徵重要度排行 (前 5 名)：")
importances = dt_model.feature_importances_
feature_imp_df = pd.DataFrame({"Feature": X_train.columns, "Importance": importances})
feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=False)
print(feature_imp_df.head(5).to_string(index=False))

# 6. 儲存模型到 models/ 目錄
os.makedirs("models", exist_ok=True)
model_output_path = "models/decision_tree.joblib"
joblib.dump(dt_model, model_output_path)
print(f"\n💾 決策樹模型已成功儲存至：{model_output_path}")