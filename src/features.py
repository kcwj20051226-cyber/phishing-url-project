import re
from urllib.parse import urlparse


def extract_features(url):
  """輸入一個 URL 字串，萃取出 17 個數值特徵"""
  if not isinstance(url, str):
    url = str(url)

  features = {}

  # 1. 網址總長度
  features["url_length"] = len(url)

  # 解析 URL 結構
  try:
    parsed = urlparse(url)
    if not parsed.scheme and not parsed.netloc:
      parsed = urlparse("http://" + url)
    netloc = parsed.netloc
  except Exception:
    netloc = ""

  # 2. 網域（Domain）長度
  domain = netloc.split(":")[0]  # 去除可能帶有的 port
  features["domain_length"] = len(domain)

  # 3. 子網域數量 (例如 a.b.example.com -> 有 2 個子網域)
  parts = domain.split(".")
  features["subdomain_count"] = max(0, len(parts) - 2)

  # 4~8. 各種特殊符號計算
  features["dot_count"] = url.count(".")
  features["dash_count"] = url.count("-")
  features["question_count"] = url.count("?")
  features["equal_count"] = url.count("=")
  features["at_count"] = url.count("@")

  # 9. 數字比例 (數字字元數 / 總長度)
  digits = sum(c.isdigit() for c in url)
  features["digit_ratio"] = digits / len(url) if len(url) > 0 else 0.0

  # 10. 特殊符號比例 (非英數字元數 / 總長度)
  special_chars = sum(not c.isalnum() for c in url)
  features["special_char_ratio"] = (
      special_chars / len(url) if len(url) > 0 else 0.0
  )

  # 11. 是否直接使用 IP 當網域 (常見於惡意連結)
  ip_pattern = r"\b\d{1,3}(\.\d{1,3}){3}\b"
  features["has_ip"] = 1 if re.search(ip_pattern, domain) or re.search(ip_pattern, url) else 0

  # 12. 是否使用 HTTPS (正常網站多數為 1，釣魚常為 0 或 http)
  features["has_https"] = 1 if parsed.scheme == "https" else 0

  # 13~17. 可疑關鍵字檢測
  url_lower = url.lower()
  features["has_login_word"] = 1 if "login" in url_lower else 0
  features["has_verify_word"] = 1 if "verify" in url_lower else 0
  features["has_secure_word"] = 1 if "secure" in url_lower else 0
  features["has_account_word"] = 1 if "account" in url_lower else 0
  features["has_update_word"] = 1 if "update" in url_lower else 0

  return features


# --- 測試區塊（直接執行這支檔案可以印出測試結果） ---
if __name__ == "__main__":
  test_url = "http://paypal-login-security-check.com/verify?id=12345"
  print(f"測試網址: {test_url}")
  print("萃取出的特徵結果：")
  res = extract_features(test_url)
  for k, v in res.items():
    print(f"  - {k}: {v}")