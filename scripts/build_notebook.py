"""Generate the full evaluation notebook."""

import json
from pathlib import Path


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


cells = [
    md_cell("# Credit Risk Model — Full Evaluation\n\nPD modeling with Logistic Regression & XGBoost, regulatory metrics, and stress testing."),
    code_cell("""import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from xgboost import XGBClassifier

ROOT = Path.cwd()
if not (ROOT / 'ML').exists():
    ROOT = ROOT.parent.parent if ROOT.name == 'notebooks' else ROOT.parent
sys.path.insert(0, str(ROOT))

from ML.features.engineering import clean_raw_data, engineer_features, prepare_training_frame
from ML.features.preprocessing import fit_preprocessor
from ML.utils import compute_metrics
from ML.risk_scoring import enrich_prediction, compute_ecl
from backend.data.macro_loader import load_macro_scenarios
from backend.stress.scenarios import apply_pd_shock

DATA_PATH = ROOT / 'data' / 'credit_risk_dataset_new.csv'
print('Project root:', ROOT)"""),
    md_cell("## 1. Exploratory Data Analysis"),
    code_cell("""df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)
print('Default rate:', f"{df['loan_status'].mean():.2%}")
df.head()"""),
    code_cell("df.info()\ndf.describe()"""),
    code_cell("""print('Missing values:\\n', df.isnull().sum())
print('Age > 100:', (df['person_age'] > 100).sum())
print('Emp length > 50:', (df['person_emp_length'] > 50).sum())
print('Max age:', df['person_age'].max())"""),
    code_cell("""fig, axes = plt.subplots(2, 2, figsize=(12, 8))
sns.histplot(df['person_age'], kde=True, ax=axes[0,0]); axes[0,0].set_title('Age Distribution')
sns.histplot(df['person_income'], kde=True, ax=axes[0,1]); axes[0,1].set_title('Income Distribution')
sns.countplot(data=df, x='loan_grade', hue='loan_status', ax=axes[1,0]); axes[1,0].set_title('Default by Grade')
sns.boxplot(data=df, x='loan_status', y='loan_percent_income', ax=axes[1,1]); axes[1,1].set_title('DTI vs Default')
plt.tight_layout(); plt.show()"""),
    md_cell("## 2. Data Cleaning & Feature Engineering"),
    code_cell("""df_clean = clean_raw_data(df)
X, y, feature_cols = prepare_training_frame(df_clean)
print('Features:', feature_cols)
print('Target distribution:\\n', y.value_counts(normalize=True))"""),
    md_cell("## 3. Train/Test Split & Preprocessing"),
    code_cell("""X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
preprocessor = fit_preprocessor(X_train)
X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)
pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
print('Train:', len(X_train), 'Test:', len(X_test))"""),
    md_cell("## 4. Model Training — Logistic Regression & XGBoost"),
    code_cell("""lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train_t, y_train)
lr_prob = lr.predict_proba(X_test_t)[:, 1]

xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight=pos_weight,
                    eval_metric='logloss', random_state=42, n_jobs=-1)
xgb.fit(X_train_t, y_train)
xgb_prob = xgb.predict_proba(X_test_t)[:, 1]
print('Models trained.')"""),
    md_cell("## 5. Model Metrics"),
    code_cell("""lr_metrics = compute_metrics(y_test.values, lr_prob)
xgb_metrics = compute_metrics(y_test.values, xgb_prob)
metrics_df = pd.DataFrame([lr_metrics, xgb_metrics], index=['Logistic Regression', 'XGBoost'])
metrics_df"""),
    md_cell("## 6. Confusion Matrix"),
    code_cell("""for name, prob in [('LR', lr_prob), ('XGB', xgb_prob)]:
    cm = confusion_matrix(y_test, (prob >= 0.5).astype(int))
    print(f'\\n{name} Confusion Matrix:')
    print(cm)
    print(classification_report(y_test, (prob >= 0.5).astype(int)))"""),
    md_cell("## 7. ROC Curve"),
    code_cell("""fig = go.Figure()
for name, prob, color in [('LR', lr_prob, 'blue'), ('XGB', xgb_prob, 'red')]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC={auc(fpr,tpr):.3f})', line=dict(color=color)))
fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR')
fig.show()"""),
    md_cell("## 8. Feature Importance"),
    code_cell("""importance = pd.Series(xgb.feature_importances_, index=feature_cols).sort_values(ascending=True).tail(15)
fig = px.bar(x=importance.values, y=importance.index, orientation='h', title='XGBoost Feature Importance (Top 15)')
fig.show()"""),
    md_cell("## 9. Lift Chart"),
    code_cell("""def lift_chart(y_true, y_prob, n_bins=10):
    data = pd.DataFrame({'y': y_true, 'prob': y_prob})
    data['bin'] = pd.qcut(data['prob'], n_bins, duplicates='drop')
    lift = data.groupby('bin', observed=True)['y'].agg(['mean', 'count'])
    lift['lift'] = lift['mean'] / data['y'].mean()
    return lift.reset_index()

lift = lift_chart(y_test.values, xgb_prob)
fig = px.bar(lift, x='bin', y='lift', title='Lift Chart (XGBoost)', labels={'lift': 'Lift'})
fig.show()"""),
    md_cell("## 10. Stress Test Examples"),
    code_cell("""scenarios = load_macro_scenarios()
for name, s in scenarios.items():
    print(f"{name}: portfolio PD multiplier = {s.portfolio_pd_multiplier:.2f}")

sample = df_clean.sample(200, random_state=42)
sample_probs = xgb.predict_proba(preprocessor.transform(prepare_training_frame(sample)[0]))[:, 1]

stress_rows = []
for name, scenario in scenarios.items():
    shocked = apply_pd_shock(sample_probs, scenario)
    total_ecl = 0.0
    for p, ead in zip(shocked, sample['loan_amnt']):
        stage = enrich_prediction(float(p), float(ead))['ifrs9_stage']
        total_ecl += compute_ecl(float(p), float(ead), stage)
    stress_rows.append({'scenario': name, 'avg_pd': float(shocked.mean()), 'total_ecl': total_ecl})

pd.DataFrame(stress_rows)"""),
    code_cell("""stress_df = pd.DataFrame(stress_rows)
fig = px.bar(stress_df, x='scenario', y=['avg_pd', 'total_ecl'], barmode='group', title='Stress Test Comparison')
fig.show()"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent.parent / "docs" / "notebooks" / "credit_risk_model_full_evaluation.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"Wrote {out}")
