import os
import pandas as pd

# 1. 讀取原始大檔案
input_path = "data/raw/phishing_site_urls.csv"
if not os.path.exists(input_path):
  print(f"找不到檔案：{input_path}，請確認是否已放入資料！")
  exit()

print("正在讀取並處理資料...")
df = pd.read_csv(input_path)

# 2. 處理正常網址 (標籤設為 0)
legitimate_df = df[df["Label"] == "good"][["URL"]].copy()
legitimate_df.rename(columns={"URL": "url"}, inplace=True)
legitimate_df["label"] = 0  # 正常網址 = 0

# 3. 處理釣魚網址 (標籤設為 1)
phishing_df = df[df["Label"] == "bad"][["URL"]].copy()
phishing_df.rename(columns={"URL": "url"}, inplace=True)
phishing_df["label"] = 1  # 釣魚網址 = 1

# 4. 各取前 5000 筆（控制期中專案的資料量，讓跑程式的速度變快）
legitimate_df = legitimate_df.head(5000)
phishing_df = phishing_df.head(5000)

# 5. 輸出成獨立的 CSV 檔案
os.makedirs("data/raw", exist_ok=True)
legitimate_df.to_csv("data/raw/legitimate_urls.csv", index=False)
phishing_df.to_csv("data/raw/phishing_urls.csv", index=False)

# 6. （額外建議）順便合併成一個總資料集，後面訓練模型會更方便！
combined_df = pd.concat([legitimate_df, phishing_df], ignore_index=True)
# 隨機打亂順序，避免模型學到順序性偏差
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
combined_df.to_csv("data/raw/phishing_dataset.csv", index=False)

print("---")
print(f"✅ 正常網址 ({len(legitimate_df)} 筆，label=0) -> data/raw/legitimate_urls.csv")
print(f"✅ 釣魚網址 ({len(phishing_df)} 筆，label=1) -> data/raw/phishing_urls.csv")
print(f"✅ 合併總資料集 ({len(combined_df)} 筆) -> data/raw/phishing_dataset.csv")