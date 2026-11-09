"""V4: PhiUSIIL HTML feature baseline. Offline input only; not a safety verdict."""
import argparse
import json
from html.parser import HTMLParser
from pathlib import Path

FEATURES = ['HasTitle', 'NoOfiFrame', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'NoOfImage']
MODEL_PATH = Path('results/v4_html_model.joblib')

class FeatureParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.counts = dict.fromkeys(FEATURES, 0)
        self.in_title = False
        self.title_text = []
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        tag = tag.lower()
        if tag == 'title': self.in_title = True
        if tag == 'iframe': self.counts['NoOfiFrame'] += 1
        if tag == 'img': self.counts['NoOfImage'] += 1
        if tag == 'input':
            kind = (attrs.get('type') or 'text').lower()
            if kind == 'password': self.counts['HasPasswordField'] = 1
            if kind == 'hidden': self.counts['HasHiddenFields'] = 1
            if kind in ('submit','image'): self.counts['HasSubmitButton'] = 1
        if tag == 'button' and (attrs.get('type') or 'submit').lower() == 'submit':
            self.counts['HasSubmitButton'] = 1
    def handle_endtag(self, tag):
        if tag.lower() == 'title': self.in_title = False
    def handle_data(self, data):
        if self.in_title: self.title_text.append(data)
    def result(self):
        self.counts['HasTitle'] = int(bool(''.join(self.title_text).strip()))
        return self.counts

def train():
    import joblib
    import pandas as pd
    from ucimlrepo import fetch_ucirepo
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    print('正在下載 UCI PhiUSIIL (ID 967)，首次需要網路，資料較大...', flush=True)
    ds = fetch_ucirepo(id=967)
    X = ds.data.features
    missing = [c for c in FEATURES if c not in X.columns]
    if missing: raise RuntimeError(f'資料集缺少特徵：{missing}；實際欄位：{list(X.columns)}')
    X = X[FEATURES].apply(pd.to_numeric, errors='coerce')
    yraw = pd.to_numeric(ds.data.targets.iloc[:, 0], errors='coerce')
    # UCI 967: 1 legitimate, 0 phishing. Model: 1 phishing, 0 legitimate.
    data = X.copy()
    data['target'] = (yraw == 0).astype(int)
    data = data.loc[yraw.isin([0,1])].dropna()
    X, y = data[FEATURES], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, min_samples_leaf=3, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    print(f'總筆數：{len(data)} / 訓練：{len(X_train)} / 測試：{len(X_test)}')
    print('Accuracy:', round(accuracy_score(y_test,pred),4))
    print('分類報告（1=釣魚）：\n',classification_report(y_test,pred, target_names=['正常','釣魚'], zero_division=0))
    print('混淆矩陣（列真實，欄預測，0正常1釣魚）：\n', confusion_matrix(y_test,pred))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model':clf,'features':FEATURES,'dataset':'UCI PhiUSIIL #967','label_mapping':'1 phishing, 0 legitimate'}, MODEL_PATH)
    print('模型已儲存：', MODEL_PATH)
    print('注意：來源資料預先計算的特徵與本程式 HTML 提取實作可能有差異；評估成績不可當成新 HTML 的實際準確率。')

def predict(html_path):
    import joblib
    import pandas as pd
    if not MODEL_PATH.exists():
        raise SystemExit('找不到模型。請先執行：py v4_html_ai.py train')
    html_path = Path(html_path)
    if not html_path.is_file(): raise SystemExit(f'找不到 HTML 檔案：{html_path}')
    if html_path.stat().st_size > 2_000_000: raise SystemExit('HTML 檔案超過 2MB 限制')
    p = FeatureParser()
    p.feed(html_path.read_text(encoding='utf-8-sig', errors='replace'))
    features = p.result()
    artifact = joblib.load(MODEL_PATH) # Only load locally trusted files
    features_in_order = artifact['features']
    row = pd.DataFrame([[features[c] for c in features_in_order]],columns=features_in_order)
    prob = float(artifact['model'].predict_proba(row)[0][list(artifact['model'].classes_).index(1)])
    print(json.dumps({'html_file':str(html_path),'features':features,'experimental_phishing_class_score':round(prob,4), 'note':'僅供研究的模型分數，不是可信的風險機率或安全判定。HTML 特徵只有六個，且不同資料集特徵提取定義可能不同。'},ensure_ascii=False,indent=2))

def main():
    ap = argparse.ArgumentParser(description='V4 HTML 特徵 AI 原型，僅離線分析')
    sub = ap.add_subparsers(dest='command',required=True)
    sub.add_parser('train',help='下載公開資料集並訓練')
    p = sub.add_parser('predict',help='預測本機 HTML'); p.add_argument('html_file')
    args = ap.parse_args()
    if args.command == 'train': train()
    else: predict(args.html_file)

if __name__ == '__main__': main()
