
import streamlit as st
import joblib
import pandas as pd
import os
import sys
import json
import tempfile
import altair as alt

from pathlib import Path
from urllib.parse import urlsplit

# =========================================================
# 1. 專案路徑與模組
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.features import extract_features
from src.validate_html_features import inspect


def project_path(*parts):
    return BASE_DIR.joinpath(*parts)


# =========================================================
# 2. 頁面設定
# =========================================================

st.set_page_config(
    page_title="智慧型釣魚網站偵測系統",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. 模型設定
# =========================================================

URL_MODELS = {
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "Logistic Regression": "models/logistic_regression.pkl"
}

HTML_MODELS = {
    "v4 原始六特徵": "models/html_rf_v4.joblib",
    "v4.2 六特徵": "models/html_rf_v4_2_six_features.joblib",
    "v4.2 五特徵": "models/html_rf_v4_2_five_features.joblib"
}


@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_html_model(path):
    if os.path.exists(path):
        artifact = joblib.load(path)

        if not isinstance(artifact, dict):
            raise ValueError("HTML 模型格式錯誤")

        if "model" not in artifact or "features" not in artifact:
            raise ValueError("HTML 模型缺少必要欄位")

        return artifact

    return None


# =========================================================
# 4. HTML 靜態分析
# =========================================================

def analyze_html(html_content, base_url):

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            encoding="utf-8",
            delete=False
        ) as temp_file:

            temp_file.write(html_content)
            temp_path = Path(temp_file.name)

        result = inspect(temp_path, base_url)

        result["file"] = "使用者提供的 HTML"
        result["base_url"] = base_url

        return result

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# =========================================================
# 5. HTML 機器學習預測
# =========================================================

def predict_html_model(result, artifact):

    model = artifact["model"]
    features = artifact["features"]

    html_features = result["model_six_features"]

    missing = [
        feature for feature in features
        if feature not in html_features
    ]

    if missing:
        raise ValueError(f"HTML 特徵缺失：{missing}")

    input_df = pd.DataFrame(
        [[html_features[name] for name in features]],
        columns=features
    )

    if hasattr(model, "feature_names_in_"):
        input_df = input_df.reindex(
            columns=list(model.feature_names_in_)
        )

    prediction = int(model.predict(input_df)[0])

    classes = list(model.classes_)
    phishing_index = classes.index(1)

    score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        score = float(probabilities[phishing_index] * 100)

    return {
        "prediction": prediction,
        "label": "疑似釣魚網站" if prediction == 1 else "模型判定正常",
        "phishing_score": score,
        "features": features
    }


# =========================================================
# 6. HTML 分析報告
# =========================================================

def display_html_report(result, predictions):

    st.markdown("---")
    st.header("📋 HTML 安全分析報告")

    # HTML 機器學習結果
    st.subheader("🤖 HTML 機器學習模型預測")

    st.caption(
        "0 = Legitimate，1 = Phishing。"
        "模型分數不是經校準的真實風險機率。"
    )

    model_columns = st.columns(3)

    for index, (name, output) in enumerate(predictions.items()):

        with model_columns[index]:

            st.markdown(f"**{name}**")

            if "error" in output:
                st.error(output["error"])
                continue

            if output["prediction"] == 1:
                st.error("🚨 疑似釣魚網站")
            else:
                st.success("✅ 模型判定正常")

            if output["phishing_score"] is not None:
                st.metric(
                    "釣魚類別模型分數",
                    f"{output['phishing_score']:.2f}%"
                )

            st.caption(
                f"使用特徵：{len(output['features'])} 個"
            )

    st.info(
        "以上為研究性模型預測結果，"
        "不能直接視為新網站的實際安全機率。"
    )

    # HTML 六項特徵
    st.markdown("---")
    st.subheader("🔍 HTML 基本特徵")

    features = result["model_six_features"]

    c1, c2, c3 = st.columns(3)

    c1.metric("iframe 數量", features["NoOfiFrame"])
    c2.metric("圖片數量", features["NoOfImage"])
    c3.metric(
        "密碼欄位",
        "有" if features["HasPasswordField"] else "無"
    )

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "網頁標題",
        "有" if features["HasTitle"] else "無"
    )

    c5.metric(
        "提交按鈕",
        "有" if features["HasSubmitButton"] else "無"
    )

    c6.metric(
        "隱藏欄位",
        "有" if features["HasHiddenFields"] else "無"
    )

    # HTML 規則分析
    st.markdown("---")
    st.subheader("🚨 HTML 可疑特徵檢查")

    extra = result["additional_review_features"]

    risk_items = [
        ("隱藏 iframe", extra["HiddenIframes"]),
        ("跨網域表單", extra["CrossDomainForms"]),
        ("跨網域密碼表單", extra["CrossDomainPasswordForms"]),
        ("Meta Refresh", extra["MetaRefreshes"])
    ]

    detected = []

    for name, count in risk_items:

        if count > 0:
            st.warning(f"⚠️ {name}：發現 {count} 個")
            detected.append(name)
        else:
            st.success(f"✅ {name}：未偵測到")

    st.markdown("---")
    st.subheader("🛡️ HTML 分析摘要")

    if detected:
        st.warning(
            f"發現 {len(detected)} 類需要進一步檢查的特徵。"
        )

        for item in detected:
            st.write(f"- {item}")
    else:
        st.success("未發現目前規則所檢查的可疑 HTML 特徵。")

    st.caption(
        "靜態分析結果不能單獨證明網站安全或惡意。"
    )

    # 表單分析
    st.markdown("---")
    st.subheader("📝 表單分析")

    forms = result.get("forms", [])

    if forms:

        form_df = pd.DataFrame(forms)

        form_df = form_df.rename(
            columns={
                "action": "表單 Action",
                "destination": "提交目的地",
                "cross_domain": "是否跨網域",
                "password": "是否包含密碼欄位"
            }
        )

        st.dataframe(form_df, use_container_width=True)

    else:
        st.info("沒有發現 HTML 表單。")

    # 特徵詳情
    with st.expander("📊 查看完整 HTML 特徵"):

        all_features = {
            **features,
            **extra
        }

        feature_df = pd.DataFrame(
            list(all_features.items()),
            columns=["特徵名稱", "數值"]
        )

        st.dataframe(feature_df, use_container_width=True)

    # JSON 匯出
    st.markdown("---")
    st.subheader("📥 匯出分析報告")

    report = dict(result)
    report["html_model_predictions"] = predictions

    report_json = json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    )

    st.download_button(
        "下載完整 HTML 分析報告",
        data=report_json,
        file_name="html_security_report.json",
        mime="application/json",
        use_container_width=True
    )


# =========================================================
# 7. 側邊欄
# =========================================================

st.sidebar.title("🛡️ 智慧型釣魚網站偵測系統")

app_mode = st.sidebar.selectbox(
    "選擇功能",
    [
        "🔍 URL 即時偵測",
        "🧩 HTML 網頁安全分析",
        "📊 模型效能與實驗比較",
        "📖 專案介紹"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 URL 預測模型")

url_model_choice = st.sidebar.selectbox(
    "選擇 URL 模型",
    list(URL_MODELS.keys())
)


# =========================================================
# 頁面一：URL 機器學習
# =========================================================

if app_mode == "🔍 URL 即時偵測":

    st.title("🔍 URL 釣魚網站即時偵測")

    st.markdown(
        "使用機器學習分析網址結構與特徵，"
        "判斷是否具有釣魚網站風險。"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        url_input = st.text_input(
            "輸入網址",
            placeholder="https://example.com/login"
        )

    with col2:
        st.write("")
        st.write("")

        submit = st.button(
            "開始 URL 分析 🚀",
            type="primary",
            use_container_width=True
        )

    if submit:

        if not url_input.strip():
            st.warning("請輸入網址。")

        else:

            model_path = project_path(
                URL_MODELS[url_model_choice]
            )

            model = load_model(str(model_path))

            if model is None:
                st.error(f"找不到模型：{model_path}")

            else:

                try:

                    with st.spinner("正在分析 URL..."):

                        features = extract_features(
                            url_input.strip()
                        )

                        input_df = pd.DataFrame([features])

                        if hasattr(model, "feature_names_in_"):

                            missing = [
                                f for f in model.feature_names_in_
                                if f not in input_df.columns
                            ]

                            if missing:
                                raise ValueError(
                                    f"缺少模型需要的特徵：{missing}"
                                )

                            input_df = input_df[
                                model.feature_names_in_
                            ]

                        prediction = int(
                            model.predict(input_df)[0]
                        )

                        score = None

                        if hasattr(model, "predict_proba"):

                            probabilities = model.predict_proba(
                                input_df
                            )[0]

                            class_index = list(
                                model.classes_
                            ).index(prediction)

                            score = float(
                                probabilities[class_index] * 100
                            )

                    st.markdown("---")
                    st.subheader("📋 URL 分析結果")

                    c1, c2 = st.columns(2)

                    with c1:

                        if prediction == 1:
                            st.error("🚨 疑似釣魚網址")
                        else:
                            st.success("✅ 模型判定正常")

                        if score is not None:
                            st.metric(
                                "模型預測分數",
                                f"{score:.2f}%"
                            )

                    with c2:

                        st.info(
                            f"使用模型：{url_model_choice}"
                        )

                        st.metric(
                            "網址長度",
                            len(url_input)
                        )

                    with st.expander("🔍 URL 特徵詳情"):

                        feature_df = pd.DataFrame(
                            list(features.items()),
                            columns=["特徵", "數值"]
                        )

                        st.dataframe(
                            feature_df,
                            use_container_width=True
                        )

                except Exception as e:
                    st.error(f"URL 分析失敗：{e}")


# =========================================================
# 頁面二：HTML 機器學習 + 靜態分析
# =========================================================

elif app_mode == "🧩 HTML 網頁安全分析":

    st.title("🧩 HTML 網頁安全分析系統")

    st.markdown(
        "結合 HTML Random Forest 機器學習模型"
        "與 HTML 靜態特徵檢查。"
    )

    st.markdown("---")

    input_mode = st.radio(
        "選擇 HTML 輸入方式",
        [
            "📄 上傳 HTML 檔案",
            "📝 貼上 HTML 原始碼"
        ],
        horizontal=True
    )

    html_content = ""

    if input_mode == "📄 上傳 HTML 檔案":

        uploaded_file = st.file_uploader(
            "選擇 HTML 檔案",
            type=["html", "htm"]
        )

        if uploaded_file is not None:

            file_bytes = uploaded_file.getvalue()

            if len(file_bytes) > 2 * 1024 * 1024:
                st.error("檔案不可超過 2 MB。")
            else:
                html_content = file_bytes.decode(
                    "utf-8-sig",
                    errors="replace"
                )

                st.success(
                    f"已載入：{uploaded_file.name}"
                )

    else:

        html_content = st.text_area(
            "HTML 原始碼",
            height=250,
            placeholder="<html>...</html>"
        )

    st.markdown("---")

    base_url = st.text_input(
        "HTML 原始網站網址",
        value="https://demo.example",
        help="只用於跨網域表單比較，不會自動連線抓取網站。"
    )

    st.caption(
        "目前只分析本機 HTML 或使用者貼上的 HTML。"
    )

    analyze_btn = st.button(
        "🔍 開始完整 HTML 分析",
        type="primary",
        use_container_width=True
    )

    if analyze_btn:

        parsed_url = urlsplit(base_url.strip())

        if not html_content.strip():
            st.warning("請先提供 HTML 內容。")

        elif len(html_content.encode("utf-8")) > 2 * 1024 * 1024:
            st.error("HTML 內容不可超過 2 MB。")

        elif (
            parsed_url.scheme not in ("http", "https")
            or not parsed_url.hostname
        ):
            st.error("請輸入有效的網站網址。")

        else:

            try:

                with st.spinner(
                    "正在執行 HTML 機器學習與靜態分析..."
                ):

                    result = analyze_html(
                        html_content,
                        base_url.strip()
                    )

                    predictions = {}

                    for name, relative_path in HTML_MODELS.items():

                        path = project_path(relative_path)

                        try:

                            artifact = load_html_model(str(path))

                            if artifact is None:

                                predictions[name] = {
                                    "error": f"找不到模型：{relative_path}"
                                }

                                continue

                            predictions[name] = predict_html_model(
                                result,
                                artifact
                            )

                        except Exception as e:

                            predictions[name] = {
                                "error": str(e)
                            }

                    st.session_state["html_result"] = result
                    st.session_state["html_predictions"] = predictions

            except Exception as e:

                st.error(f"HTML 分析失敗：{e}")

                st.session_state.pop("html_result", None)
                st.session_state.pop("html_predictions", None)

    if "html_result" in st.session_state:

        display_html_report(
            st.session_state["html_result"],
            st.session_state["html_predictions"]
        )


# =========================================================
# 頁面三：模型效能與實驗比較
# =========================================================

elif app_mode == "📊 模型效能與實驗比較":

    st.title("📊 模型效能與實驗比較")

    # URL 模型
    st.subheader("🌐 URL 模型效能")

    url_csv = project_path(
        "outputs",
        "tables",
        "model_comparison.csv"
    )

    if url_csv.exists():

        url_df = pd.read_csv(url_csv)

        st.dataframe(
            url_df,
            use_container_width=True
        )

    else:
        st.warning("找不到 URL 模型比較資料。")

    st.markdown("---")

    # HTML 特徵比較
    st.subheader("🧩 HTML 特徵消融實驗")

    html_csv = project_path(
        "outputs",
        "tables",
        "html_feature_comparison.csv"
    )

    if html_csv.exists():

        html_df = pd.read_csv(html_csv)

        st.dataframe(
            html_df,
            use_container_width=True
        )

        metric_columns = [
            "Accuracy",
            "Precision_Phishing",
            "Recall_Phishing",
            "F1_Phishing"
        ]

        if all(
            column in html_df.columns
            for column in metric_columns
        ):

            # 轉換成百分比
            chart_df = html_df[
                ["Model"] + metric_columns
            ].copy()

            chart_df[metric_columns] = (
                chart_df[metric_columns] * 100
            )

            # 轉換成長格式
            chart_long = chart_df.melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score"
            )

            metric_names = {
                "Accuracy": "Accuracy 準確率",
                "Precision_Phishing": "Precision 精確率",
                "Recall_Phishing": "Recall 召回率",
                "F1_Phishing": "F1-score"
            }

            chart_long["Metric"] = (
                chart_long["Metric"].map(metric_names)
            )

            st.subheader(
                "📈 五特徵與六特徵效能比較"
            )

            # 群組長條圖：四個指標並排，不再堆疊
            chart = alt.Chart(chart_long).mark_bar().encode(

                x=alt.X(
                    "Model:N",
                    title="模型",
                    axis=alt.Axis(labelAngle=0)
                ),

                xOffset=alt.XOffset("Metric:N"),

                y=alt.Y(
                    "Score:Q",
                    title="效能指標 (%)",
                    scale=alt.Scale(domain=[0, 100])
                ),

                color=alt.Color(
                    "Metric:N",
                    title="評估指標"
                ),

                tooltip=[
                    alt.Tooltip("Model:N", title="模型"),
                    alt.Tooltip("Metric:N", title="指標"),
                    alt.Tooltip(
                        "Score:Q",
                        title="分數",
                        format=".2f"
                    )
                ]

            ).properties(
                height=420
            )

            st.altair_chart(
                chart,
                use_container_width=True
            )

            st.info(
                "六特徵模型在這次相同測試集實驗中，"
                "Accuracy 與 F1-score 均高於五特徵模型。"
            )

    else:
        st.warning("找不到 HTML 特徵比較資料。")

    # URL 圖表
    st.markdown("---")
    st.subheader("📈 URL 模型視覺化")

    figures = [
        (
            "模型效能比較",
            "outputs/figures/model_comparison.png"
        ),
        (
            "隨機森林混淆矩陣",
            "outputs/figures/confusion_matrix_rf.png"
        ),
        (
            "特徵重要度",
            "outputs/figures/feature_importance_rf.png"
        ),
        (
            "資料集類別分布",
            "outputs/figures/label_distribution.png"
        )
    ]

    col1, col2 = st.columns(2)

    for index, (title, filename) in enumerate(figures):

        target_col = col1 if index % 2 == 0 else col2

        with target_col:

            st.markdown(f"**{title}**")

            image_path = project_path(filename)

            if image_path.exists():

                st.image(
                    str(image_path),
                    use_container_width=True
                )

            else:
                st.info("目前沒有此圖表。")


# =========================================================
# 頁面四：專案介紹
# =========================================================

elif app_mode == "📖 專案介紹":

    st.title("📖 智慧型釣魚網站偵測系統")

    st.markdown(
        """
        ## 🎯 專案目標

        結合網址機器學習與 HTML 網頁分析，
        建立多層次的釣魚網站風險偵測系統。

        ## 🌐 第一部分：URL 機器學習

        - Random Forest
        - Decision Tree
        - Logistic Regression
        - URL 特徵萃取
        - 模型效能比較

        ## 🧩 第二部分：HTML 機器學習

        - Random Forest 六特徵模型
        - Random Forest 五特徵模型
        - 五特徵與六特徵實驗比較
        - HTML 模型預測

        ## 🔍 第三部分：HTML 靜態分析

        - 隱藏 iframe 偵測
        - 跨網域表單分析
        - 跨網域密碼表單
        - Meta Refresh 偵測
        - HTML 分析報告

        ## 🔬 研究方向

        探討 URL 特徵與 HTML 特徵在釣魚網站
        偵測上的互補性，作為後續特徵融合
        與綜合風險判斷的研究基礎。
        """
    )

    st.markdown("---")
    st.subheader("👥 團隊成員")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("113213021", "陳昱維")
    c2.metric("113213065", "謝承道")
    c3.metric("113213069", "蘇資晉")
    c4.metric("113213076", "王志騰")
    c5.metric("113213077", "温佳哲")
