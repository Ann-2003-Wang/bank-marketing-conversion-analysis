#src/utils.py

#提供数据加载、缺失值检查、无穷值处理、绘图等通用函数。


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score

def load_raw_data():
    """加载原始训练和测试数据"""
    from src.config import TRAIN_RAW, TEST_RAW
    train = pd.read_csv(TRAIN_RAW)
    test = pd.read_csv(TEST_RAW)
    X_train = train.drop('y', axis=1)
    y_train = train['y']
    X_test = test.copy()
    return X_train, y_train, X_test

def check_missing(df, name='DataFrame'):
    """打印缺失值统计信息"""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print(f"✓ {name} 无缺失值")
    else:
        print(f"⚠ {name} 存在缺失值:\n{missing}")

def handle_infinity_with_quartiles(X_train, X_test):
    """使用训练集上下四分位数替换无穷大/无穷小"""
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if np.isinf(X_train[col]).any() or np.isinf(X_test[col]).any():
            finite = X_train[col][~np.isinf(X_train[col])]
            Q1, Q3 = finite.quantile(0.25), finite.quantile(0.75)
            X_train_clean[col] = X_train_clean[col].replace([np.inf, -np.inf], [Q3, Q1])
            X_test_clean[col] = X_test_clean[col].replace([np.inf, -np.inf], [Q3, Q1])
    return X_train_clean, X_test_clean

def handle_pdays_special_value(X_train, X_test):
    """将 pdays 中的 999 替换为 -1"""
    X_train = X_train.copy()
    X_test = X_test.copy()
    if 'pdays' in X_train.columns:
        X_train['pdays'] = X_train['pdays'].replace(999, -1)
        X_test['pdays'] = X_test['pdays'].replace(999, -1)
    return X_train, X_test

def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve', save_path=None):
    """绘制并保存 ROC 曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """绘制 LightGBM 特征重要性"""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    plt.figure(figsize=(10, 8))
    plt.title('Feature Importance')
    plt.barh(range(top_n), importance[indices][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[::-1]])
    plt.xlabel('Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()