import os
import pandas as pd
from features import extract_features


def convert_dataset_to_features(input_csv, output_csv):
  print(f"正在處理：{input_csv} -> 轉換為特徵表...")
  df = pd.read_csv(input_csv)

  feature_list = []
  for url in df["url"]:
    feature_list.append(extract_features(url))

  features_df = pd.DataFrame(feature_list)
  # 把原本的 url 跟 label 拼回最前面
  features_df.insert(0, "url", df["url"].values)
  features_df["label"] = df["label"].values

  os.makedirs(os.path.dirname(output_csv), exist_ok=True)
  features_df.to_csv(output_csv, index=False)
  print(f"✅ 已成功輸出至：{output_csv} (共 {len(features_df)} 筆)")


# 批次處理 train, valid, test
convert_dataset_to_features(
    "data/processed/train.csv", "data/processed/train_features.csv"
)
convert_dataset_to_features(
    "data/processed/valid.csv", "data/processed/valid_features.csv"
)
convert_dataset_to_features(
    "data/processed/test.csv", "data/processed/test_features.csv"
)

print("\n🎉 第 6 步特徵表轉換全部完成！")