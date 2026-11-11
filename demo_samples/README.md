# HTML Demo 測試案例（模型驗證版）

使用 `https://example.com/` 作為 HTML 原始網站網址，上傳至 Streamlit 的 HTML 分析頁。

| 檔案 | v4 六特徵 | v4.2 六特徵 | v4.2 五特徵 | 靜態檢查 |
|---|---|---|---|---|
| 01_normal_demo.html | 正常 | 正常 | 正常 | 四項無警示 |
| 02_phishing_image_feature_demo.html | 釣魚 | 釣魚 | 正常 | 模擬釣魚案例 |
| 03_phishing_warnings_demo.html | 釣魚 | 釣魚 | 釣魚 | 隱藏 iframe、跨網域表單、跨網域密碼表單、Meta Refresh |

這些都是離線人工模擬的 HTML。只用於展示模型反應與靜態結構檢查，不能宣稱為真實遭檢舉的惡意網站。所有表單均停用提交，並使用範例網域，不會真的送出密碼。

結果由 `src/validate_html_features.py` 相同規則所擷取特徵，交由三個上傳的 `.joblib` 模型進行預測驗證；scikit-learn 執行環境版本不同時仍應在 Streamlit 上做最後確認。
