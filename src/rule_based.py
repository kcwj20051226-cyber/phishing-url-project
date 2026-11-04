import os
import pandas as pd
from features import extract_features


def rule_based_predict(url):
  """根據自訂規則計算網址風險分數、回傳預測結果與判斷理由"""
  features = extract_features(url)
  score = 0
  reasons = []

  # 1. URL 長度 > 75
  if features["url_length"] > 75:
    score += 1
    reasons.append(f"URL 長度過長 ({features['url_length']} > 75): +1")

  # 2. 子網域數 > 2
  if features["subdomain_count"] > 2:
    score += 1
    reasons.append(f"子網域數量過多 ({features['subdomain_count']} > 2): +1")

  # 3. 有 @
  if features["at_count"] > 0:
    score += 2
    reasons.append("包含 '@' 符號（常具混淆性）: +2")

  # 4. 有 IP
  if features["has_ip"] == 1:
    score += 2
    reasons.append("網址直接使用 IP 位址: +2")

  # 5. 沒有 HTTPS
  if features["has_https"] == 0:
    score += 1
    reasons.append("未使用 HTTPS 加密: +1")

  # 6. 有 login 關鍵字
  if features["has_login_word"] == 1:
    score += 1
    reasons.append("包含可疑關鍵字 'login': +1")

  # 7. 有 verify 關鍵字
  if features["has_verify_word"] == 1:
    score += 1
    reasons.append("包含可疑關鍵字 'verify': +1")

  # 門檻判定：分數 >= 3 視為釣魚 (1)，否則正常 (0)
  prediction = 1 if score >= 3 else 0
  return score, prediction, reasons


# --- 1. 測試單一網址與理由輸出 ---
if __name__ == "__main__":
  test_url = "http://paypal-login-security-check.com/verify?id=12345"
  score, pred, reasons = rule_based_predict(test_url)
  print(f"測試網址: {test_url}")
  print(f"計算總分: {score} 分 -> 判定結果: {'釣魚 (1)' if pred == 1 else '正常 (0)'}")
  print("判斷理由：")
  for r in reasons:
    print(f"  - {r}")

  # --- 2. 在測試集 (test.csv) 上全面評估 Baseline 表現 ---
  test_csv = "data/processed/test.csv"
  if os.path.exists(test_csv):
    print("\n" + "=" * 40)
    print("正在評估 Baseline (規則法) 在測試集上的整體表現...")
    df = pd.read_csv(test_csv)

    y_true = df["label"].values
    y_pred = []

    for url in df["url"]:
      _, pred, _ = rule_based_predict(url)
      y_pred.append(pred)

    from sklearn.metrics import accuracy_score, classification_report

    acc = accuracy_score(y_true, y_pred)
    print(f"📊 Baseline 準確率 (Accuracy): {acc:.4f}")
    print("\n詳細分類報告：")
    print(
        classification_report(
            y_true, y_pred, target_names=["正常 (0)", "釣魚 (1)"]
        )
    )