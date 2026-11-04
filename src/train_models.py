import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("🚀 開始進行多模型訓練與評估比較...")

# 1. 載入前面處理好的訓練集與測試集特徵
train_path = "data/processed/train_features.csv"
test_path = "data/processed/test_features.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    print("❌ 找不到訓練集或測試集特徵檔案，請先確認 data/processed/ 底下是否有對應檔案！")
    exit()

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 分離特徵 (X) 與標籤 (y)，排除 url 欄位
X_train = train_df.drop(columns=["url", "label"])
y_train = train_df["label"]

X_test = test_df.drop(columns=["url", "label"])
y_test = test_df["label"]

# 2. 定義要訓練的模型清單 (加入隨機森林與邏輯斯迴歸)
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}

results = []

# 確保輸出資料夾存在
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

# 3. 迴圈訓練與評估每個模型
for name, model in models.items():
    print(f"  正在訓練模型: {name} ...")
    # 訓練
    model.fit(X_train, y_train)
    
    # 預測測試集
    y_pred = model.predict(X_test)
    
    # 計算各項評估指標
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-score": round(f1, 4)
    })
    
    # 儲存模型為 .pkl 格式至 models/ 資料夾
    file_name = name.lower().replace(" ", "_")
    model_path = f"models/{file_name}.pkl"
    joblib.dump(model, model_path)
    print(f"    -> 已成功儲存模型至 {model_path}")

# 4. 產出模型比較表並存入 outputs/tables/
comparison_df = pd.DataFrame(results)
comparison_path = "outputs/tables/model_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

print("\n📊 模型效能比較結果總覽：")
print("=" * 60)
print(comparison_df.to_string(index=False))
print("=" * 60)
print(f"\n💾 比較表已成功儲存至: {comparison_path}")