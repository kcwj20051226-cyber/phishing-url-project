import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. 讀取前面第 3 步產出的乾淨資料集
input_path = "data/processed/dataset_clean.csv"
if not os.path.exists(input_path):
    print(f"找不到檔案：{input_path}，請先確認第 3 步是否順利執行完成！")
    exit()

print("正在執行第 4 步：切分訓練集、驗證集與測試集...")
df = pd.read_csv(input_path)

# 2. 第一次切分：分出 70% 的訓練集 (train)，剩下 30% 暫存 (temp)
# 💡 小技巧：加上 stratify=df['label']，確保切出來的各集裡，「正常」與「釣魚」的比例都一樣均勻！
train_df, temp_df = train_test_split(
    df, test_size=0.30, random_state=42, stratify=df["label"]
)

# 3. 第二次切分：把剩下的 30% 對半切，各分 15% 給驗證集 (valid) 與測試集 (test)
valid_df, test_df = train_test_split(
    temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
)

# 4. 輸出儲存成三個獨立的 CSV 檔案
os.makedirs("data/processed", exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
valid_df.to_csv("data/processed/valid.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("---")
print("🎉 第 4 步大功告成！檔案已順利切分完畢：")
print(f"📈 訓練集 (Train) : {len(train_df)} 筆 -> data/processed/train.csv")
print(f"📊 驗證集 (Valid) : {len(valid_df)} 筆 -> data/processed/valid.csv")
print(f"🧪 測試集 (Test)  : {len(test_df)} 筆 -> data/processed/test.csv")