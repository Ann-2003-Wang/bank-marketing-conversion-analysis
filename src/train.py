import os
import warnings
from time import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, ParameterGrid, ParameterSampler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)
import lightgbm as lgb

from src.config import DATA_DIR, TEST_MS, MODEL_DIR, RANDOM_STATE, CV_FOLDS
from src.preprocessing import oversample_minority

warnings.filterwarnings('ignore')

# ==================== 路径配置 ====================
TRAIN_FILE = os.path.join(DATA_DIR, 'train_ms_processed.csv')   # 改：读未过采样训练集
TEST_FILE = TEST_MS
OUTPUT_PRED = os.path.join(DATA_DIR, 'test_predictions_best_model.csv')
OUTPUT_COMPARE = os.path.join(DATA_DIR, 'model_comparison_oof.csv')

os.makedirs(MODEL_DIR, exist_ok=True)

# 固定分类阈值
PRED_THRESHOLD = 0.5


# ==================== 数据加载 ====================
def load_data():
    """加载 processed train/test，分离特征与标签。"""
    print("=" * 60)
    print("加载 processed 数据（未过采样）...")

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)

    X_train = train.drop('y', axis=1)
    y_train = train['y'].astype(int)
    X_test = test.copy()

    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    print(f"训练集正样本比例: {y_train.mean():.4f}")
    return X_train, y_train, X_test


# ==================== 绘图函数 ====================
def plot_roc_curve(y_true, y_pred_proba, title='ROC Curve', save_path=None):
    """绘制并保存 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """绘制 LightGBM 特征重要性。"""
    if not hasattr(model, 'feature_importances_'):
        return

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


# ==================== 评估函数 ====================
def evaluate_predictions(y_true, y_pred_proba, threshold=PRED_THRESHOLD):
    """
    使用 OOF 预测结果计算评估指标。
    注意：这里评估的是“验证折拼起来的整体预测”，不是训练集内自评。
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'cv_auc': roc_auc_score(y_true, y_pred_proba),
        'cv_pr_auc': average_precision_score(y_true, y_pred_proba),
        'cv_accuracy': accuracy_score(y_true, y_pred),
        'cv_precision': precision_score(y_true, y_pred, zero_division=0),
        'cv_recall': recall_score(y_true, y_pred, zero_division=0),
        'cv_f1': f1_score(y_true, y_pred, zero_division=0),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'pred_positive_rate': float(y_pred.mean())
    }
    return metrics


# ==================== 核心：fold 内过采样 + OOF ====================
def run_oof_cv(model_builder, X, y, skf, threshold=PRED_THRESHOLD):
    """
    核心流程：
    - 用原始 processed train 做 StratifiedKFold
    - 每个 fold 内，只对训练折做过采样
    - 验证折保持原始分布，不参与过采样
    - 拼接全部验证折预测，得到 OOF 预测
    """
    oof_pred_proba = np.zeros(len(y), dtype=float)
    fold_auc_list = []

    for fold_id, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
        X_tr = X.iloc[train_idx].copy()
        y_tr = y.iloc[train_idx].copy()
        X_val = X.iloc[valid_idx].copy()
        y_val = y.iloc[valid_idx].copy()

        # 只对训练折做过采样
        X_tr_bal, y_tr_bal = oversample_minority(X_tr, y_tr)

        model = model_builder()
        model.fit(X_tr_bal, y_tr_bal)

        val_pred_proba = model.predict_proba(X_val)[:, 1]
        oof_pred_proba[valid_idx] = val_pred_proba

        fold_auc = roc_auc_score(y_val, val_pred_proba)
        fold_auc_list.append(fold_auc)

        print(f"  Fold {fold_id}/{CV_FOLDS} AUC: {fold_auc:.4f}")

    metrics = evaluate_predictions(y, oof_pred_proba, threshold=threshold)
    metrics['fold_auc_mean'] = float(np.mean(fold_auc_list))
    metrics['fold_auc_std'] = float(np.std(fold_auc_list))

    return metrics, oof_pred_proba


def fit_final_model(model_builder, X, y):
    """
    用全部训练集重新训练最终模型。
    注意：这里仍然只对“最终训练用的整张训练集”做一次过采样。
    """
    X_bal, y_bal = oversample_minority(X, y)
    model = model_builder()
    model.fit(X_bal, y_bal)
    return model


# ==================== 模型构造函数 ====================
def build_naive_bayes(var_smoothing=1e-9):
    return GaussianNB(var_smoothing=var_smoothing)


def build_knn(n_neighbors=9, weights='distance', metric='manhattan'):
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
        algorithm='auto'
    )


def build_l1_knn(max_features=20, n_neighbors=9, weights='distance', metric='manhattan'):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectFromModel(
            LogisticRegression(
                penalty='l1',
                solver='liblinear',
                C=0.1,
                random_state=RANDOM_STATE,
                max_iter=1000
            ),
            max_features=max_features
        )),
        ('knn', KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            algorithm='auto'
        ))
    ])


def build_lightgbm(
    learning_rate=0.05,
    n_estimators=150,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0
):
    return lgb.LGBMClassifier(
        objective='binary',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )


# ==================== 参数搜索 ====================
def search_best_model(model_name, model_factory, param_candidates, X, y, roc_filename=None):
    """
    手动参数搜索：
    对每组参数都跑一次“fold 内过采样 + OOF”
    再以 OOF AUC 选最优参数
    """
    print("\n" + "=" * 60)
    print(f"模型：{model_name}")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_score = -np.inf
    best_params = None
    best_metrics = None
    best_oof_pred = None

    total_candidates = len(param_candidates)

    for idx, params in enumerate(param_candidates, 1):
        print(f"\n参数组合 {idx}/{total_candidates}: {params}")

        metrics, oof_pred_proba = run_oof_cv(
            model_builder=lambda: model_factory(**params),
            X=X,
            y=y,
            skf=skf,
            threshold=PRED_THRESHOLD
        )

        print(
            f"  OOF AUC={metrics['cv_auc']:.4f}, "
            f"PR-AUC={metrics['cv_pr_auc']:.4f}, "
            f"Precision={metrics['cv_precision']:.4f}, "
            f"Recall={metrics['cv_recall']:.4f}, "
            f"F1={metrics['cv_f1']:.4f}"
        )

        if metrics['cv_auc'] > best_score:
            best_score = metrics['cv_auc']
            best_params = params
            best_metrics = metrics
            best_oof_pred = oof_pred_proba

    print(f"\n[{model_name}] 最佳参数: {best_params}")
    print(f"[{model_name}] 最佳 OOF AUC: {best_metrics['cv_auc']:.4f}")

    if roc_filename:
        plot_roc_curve(
            y_true=y,
            y_pred_proba=best_oof_pred,
            title=f'{model_name} OOF ROC Curve',
            save_path=os.path.join(MODEL_DIR, roc_filename)
        )

    # 用全部训练集重新拟合最终模型
    final_model = fit_final_model(lambda: model_factory(**best_params), X, y)

    return {
        'model_name': model_name,
        'model': final_model,
        'best_params': best_params,
        **best_metrics
    }


# ==================== 各模型训练入口 ====================
def train_naive_bayes(X_train, y_train):
    param_grid = list(ParameterGrid({
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
    }))
    return search_best_model(
        model_name='NaiveBayes',
        model_factory=build_naive_bayes,
        param_candidates=param_grid,
        X=X_train,
        y=y_train,
        roc_filename='nb_oof_roc_curve.png'
    )


def train_knn(X_train, y_train):
    param_grid = list(ParameterGrid({
        'n_neighbors': [7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }))
    return search_best_model(
        model_name='KNN',
        model_factory=build_knn,
        param_candidates=param_grid,
        X=X_train,
        y=y_train,
        roc_filename='knn_oof_roc_curve.png'
    )


def train_l1_knn(X_train, y_train):
    param_grid = list(ParameterGrid({
        'max_features': [15, 20, 25, 30],
        'n_neighbors': [7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }))
    return search_best_model(
        model_name='L1+KNN',
        model_factory=build_l1_knn,
        param_candidates=param_grid,
        X=X_train,
        y=y_train,
        roc_filename='l1_knn_oof_roc_curve.png'
    )


def train_lightgbm(X_train, y_train):
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 150, 200],
        'num_leaves': [31, 63, 127],
        'max_depth': [6, 8, -1],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }

    param_candidates = list(ParameterSampler(
        param_distributions=param_dist,
        n_iter=20,
        random_state=RANDOM_STATE
    ))

    result = search_best_model(
        model_name='LightGBM',
        model_factory=build_lightgbm,
        param_candidates=param_candidates,
        X=X_train,
        y=y_train,
        roc_filename='lgb_oof_roc_curve.png'
    )

    # 保存 LightGBM 模型和特征重要性
    if result['model_name'] == 'LightGBM':
        lgb_model = result['model']
        lgb_model.booster_.save_model(os.path.join(MODEL_DIR, 'best_lightgbm_model.txt'))
        plot_feature_importance(
            lgb_model,
            X_train.columns,
            top_n=15,
            save_path=os.path.join(MODEL_DIR, 'lgb_feature_importance.png')
        )

    return result


# ==================== 汇总对比 ====================
def generate_comparison(results):
    """输出并保存 OOF 对比结果。"""
    df = pd.DataFrame(results)

    display_cols = [
        'model_name',
        'cv_auc',
        'cv_pr_auc',
        'cv_precision',
        'cv_recall',
        'cv_f1',
        'cv_accuracy',
        'fold_auc_mean',
        'fold_auc_std',
        'tn', 'fp', 'fn', 'tp',
        'pred_positive_rate',
        'best_params'
    ]

    df = df.sort_values('cv_auc', ascending=False)

    print("\n" + "=" * 60)
    print("模型对比结果（基于 OOF）")
    print("=" * 60)
    print(df[display_cols].to_string(index=False))

    df.to_csv(OUTPUT_COMPARE, index=False)
    print(f"\n对比结果已保存至: {OUTPUT_COMPARE}")

    return df


# ==================== 测试集预测 ====================
def predict_on_test(best_model, X_test, model_name, threshold=PRED_THRESHOLD):
    """使用最优模型对测试集进行预测并保存。"""
    print("\n" + "=" * 60)
    print(f"使用最优模型 ({model_name}) 对测试集进行预测...")

    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred_label = (y_pred_proba >= threshold).astype(int)

    pred_df = pd.DataFrame({
        '样本索引': X_test.index,
        '订阅定期存款概率': y_pred_proba.round(6),
        '预测结果（0=不订阅/1=订阅）': y_pred_label
    })

    pred_df.to_csv(OUTPUT_PRED, index=False, encoding='utf-8-sig')
    print(f"预测结果已保存至: {OUTPUT_PRED}")
    print(f"总样本数: {len(pred_df)}")
    print(f"预测订阅客户数: {y_pred_label.sum()} (占比: {y_pred_label.mean():.2%})")
    print(f"预测不订阅客户数: {len(pred_df) - y_pred_label.sum()} (占比: {1 - y_pred_label.mean():.2%})")


# ==================== 主函数 ====================
def main():
    start_time = time()

    # 1. 加载数据
    X_train, y_train, X_test = load_data()

    # 2. 训练与搜索
    all_results = []

    nb_res = train_naive_bayes(X_train, y_train)
    all_results.append(nb_res)

    knn_res = train_knn(X_train, y_train)
    all_results.append(knn_res)

    l1_knn_res = train_l1_knn(X_train, y_train)
    all_results.append(l1_knn_res)

    lgb_res = train_lightgbm(X_train, y_train)
    all_results.append(lgb_res)

    # 3. 汇总比较
    df_compare = generate_comparison(all_results)

    # 4. 选择最优模型并预测测试集
    best_row = df_compare.iloc[0]
    best_model = best_row['model']
    best_name = best_row['model_name']

    predict_on_test(best_model, X_test, best_name)

    elapsed = time() - start_time
    print(f"\n全部流程完成，总耗时: {elapsed:.2f} 秒")


if __name__ == '__main__':
    main()
