import os
import pandas as pd

input_path = "data/raw/phishing_site_urls.csv"
if not os.path.exists(input_path):
    print(f"找不到檔案：{input_path}")
    exit()

print("正在執行第 3 步：資料清理與格式統一...")
df = pd.read_csv(input_path)

# 1. 確保欄位名稱乾淨並重新命名
df.columns = df.columns.str.strip()
df = df.rename(columns={"URL": "url", "Label": "label"})

# 2. 清理 url 欄位
df = df.dropna(subset=["url"])
df["url"] = df["url"].astype(str).str.strip()
df = df.drop_duplicates(subset=["url"])
df = df[(df["url"].str.len() > 3) & (df["url"].str.len() < 2000)]

# 3. 清理 label 欄位（去除空白、轉小寫）並對應成 0 與 1
df["label"] = df["label"].astype(str).str.strip().str.lower()
label_map = {"good": 0, "bad": 1}
df["label"] = df["label"].map(label_map)

# 移除對應失敗的行
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# 4. 各取 5000 筆平衡資料
legitimate_df = df[df["label"] == 0].head(5000)
phishing_df = df[df["label"] == 1].head(5000)

print(f"📊 正常網址 (label=0) 數量：{len(legitimate_df)}")
print(f"📊 釣魚網址 (label=1) 數量：{len(phishing_df)}")

# 合併並隨機打散
clean_dataset = pd.concat([legitimate_df, phishing_df], ignore_index=True)
clean_dataset = clean_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. 輸出至指定的資料夾
os.makedirs("data/processed", exist_ok=True)
output_path = "data/processed/dataset_clean.csv"
clean_dataset.to_csv(output_path, index=False)

print("---")
print(f"🎉 第 3 步大功告成！檔案已儲存至：{output_path}")
print(f"✨ 總計清理完畢：{len(clean_dataset)} 筆資料")