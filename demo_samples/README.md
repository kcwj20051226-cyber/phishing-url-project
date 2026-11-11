# HTML Demo Samples

These three files are manually prepared demonstration samples for the phishing-detection project. They are intended for local HTML analysis in the Streamlit application.

## Demo order

1. `01_normal_demo.html` — normal-looking sample; use it to demonstrate the baseline model output.
2. `02_phishing_image_feature_demo.html` — simulated phishing/login sample; use it to demonstrate the observed difference between the five-feature and six-feature models.
3. `03_phishing_warnings_demo.html` — simulated suspicious sample containing structures intended to trigger HTML static-analysis warnings.

## How to test

- Select **HTML 網頁安全分析** in the Streamlit app.
- Choose **上傳 HTML 檔案** and upload one sample at a time.
- Use `https://example.com/` as the original website URL for the cross-domain form demonstration.

These are synthetic demo samples, not verified live phishing websites. Model predictions are empirical outputs for these exact files and may differ if the feature extractor, model artifacts, or application code changes. Static warnings indicate suspicious structures for review; they do not prove maliciousness.
