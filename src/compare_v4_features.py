"""V4.2: Compare identical Random Forest settings with 6 vs 5 HTML features.
Uses UCI PhiUSIIL #967 precomputed features; results do not validate raw HTML extraction.
"""
from pathlib import Path
import csv
import json

ALL_FEATURES = ['HasTitle', 'NoOfiFrame', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'NoOfImage']
SEED = 42


def main():
    import pandas as pd
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    import joblib

    print('下載/讀取 UCI PhiUSIIL #967（235,795 筆左右）...', flush=True)
    ds = fetch_ucirepo(id=967)
    missing = [c for c in ALL_FEATURES if c not in ds.data.features.columns]
    if missing:
        raise RuntimeError(f'找不到資料欄位: {missing}')
    x = ds.data.features[ALL_FEATURES].apply(pd.to_numeric, errors='coerce')
    raw = pd.to_numeric(ds.data.targets.iloc[:, 0], errors='coerce')
    clean = x.copy()
    clean['target'] = (raw == 0).astype(int)  # UCI 原始 0=釣魚；轉成 1=釣魚
    clean = clean.loc[raw.isin([0, 1])].dropna()
    y = clean['target']
    x = clean[ALL_FEATURES]

    # Important: split row indices ONCE, then use identical train/test rows for both models.
    train_indices, test_indices = train_test_split(
        x.index.to_numpy(), test_size=0.2, random_state=SEED, stratify=y
    )
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    summary = []
    for label, columns, filename in [
        ('Six features', ALL_FEATURES, 'v4_2_six_features.joblib'),
        ('Five features (no image)', [c for c in ALL_FEATURES if c != 'NoOfImage'], 'v4_2_five_features.joblib'),
    ]:
        print(f'\n訓練 {label} ...', flush=True)
        model = RandomForestClassifier(
            n_estimators=120, min_samples_leaf=3, max_depth=12,
            random_state=SEED, n_jobs=-1
        )
        model.fit(x.loc[train_indices, columns], y.loc[train_indices])
        predictions = model.predict(x.loc[test_indices, columns])
        tn, fp, fn, tp = confusion_matrix(y.loc[test_indices], predictions, labels=[0, 1]).ravel()
        row = {
            'Model': label,
            'Features': len(columns),
            'Accuracy': round(accuracy_score(y.loc[test_indices], predictions), 4),
            'Precision_Phishing': round(precision_score(y.loc[test_indices], predictions, pos_label=1, zero_division=0), 4),
            'Recall_Phishing': round(recall_score(y.loc[test_indices], predictions, pos_label=1, zero_division=0), 4),
            'F1_Phishing': round(f1_score(y.loc[test_indices], predictions, pos_label=1, zero_division=0), 4),
            'False_Negatives': int(fn),
            'False_Positives': int(fp),
            'True_Negatives': int(tn),
            'True_Positives': int(tp),
        }
        summary.append(row)
        joblib.dump({'model': model, 'features': columns, 'label_mapping': '1 phishing; 0 legitimate', 'dataset': 'UCI PhiUSIIL #967'}, results_dir / filename)
        print(f'Accuracy: {row["Accuracy"]:.4f} | Recall: {row["Recall_Phishing"]:.4f} | F1: {row["F1_Phishing"]:.4f} | 漏判: {fn} | 誤報: {fp}')

    csv_path = results_dir / 'v4_2_feature_comparison.csv'
    with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (results_dir / 'v4_2_feature_comparison.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n====== 六特徵 vs 五特徵：相同測試資料 ======')
    print(pd.DataFrame(summary).to_string(index=False))
    print(f'\n已儲存：{csv_path}')
    print('提醒：比較的是資料集預先計算的特徵，不能直接證明對新 HTML 的安全偵測能力。')


if __name__ == '__main__':
    main()
