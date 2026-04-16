import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.utils import (
    load_raw_data,
    handle_infinity_with_quartiles,
    handle_pdays_special_value
)
from src.config import DATA_DIR, TEST_MS, TEST_QS, POS_RATIO, RANDOM_STATE


# 新增：保存“未过采样”的训练集
TRAIN_MS_PROCESSED = os.path.join(DATA_DIR, 'train_ms_processed.csv')
TRAIN_QS_PROCESSED = os.path.join(DATA_DIR, 'train_qs_processed.csv')


def fill_missing_values(X_train, X_test):
    """填充测试集中的少量缺失值；训练集如有缺失，也统一按训练集统计量填充。"""
    X_train = X_train.copy()
    X_test = X_test.copy()

    # feature_1: 中位数
    if 'feature_1' in X_train.columns:
        median_f1 = X_train['feature_1'].median()
        X_train['feature_1'] = X_train['feature_1'].fillna(median_f1)
        X_test['feature_1'] = X_test['feature_1'].fillna(median_f1)

    # feature_2: 均值
    if 'feature_2' in X_train.columns:
        mean_f2 = X_train['feature_2'].mean()
        X_train['feature_2'] = X_train['feature_2'].fillna(mean_f2)
        X_test['feature_2'] = X_test['feature_2'].fillna(mean_f2)

    # campaign: 0
    if 'campaign' in X_train.columns:
        X_train['campaign'] = X_train['campaign'].fillna(0)
        X_test['campaign'] = X_test['campaign'].fillna(0)

    # age: 修复异常值后如果产生 NaN，这里补上
    if 'age' in X_train.columns:
        age_median = X_train['age'].median()
        X_train['age'] = X_train['age'].fillna(age_median)
        X_test['age'] = X_test['age'].fillna(age_median)

    return X_train, X_test


def fix_age_outliers(X_train, X_test):
    """将年龄异常值设为 NaN，后续统一用训练集统计量填补。"""
    X_train = X_train.copy()
    X_test = X_test.copy()

    if 'age' in X_train.columns:
        mask_train = (X_train['age'] < 0) | (X_train['age'] > 100)
        mask_test = (X_test['age'] < 0) | (X_test['age'] > 100)
        X_train.loc[mask_train, 'age'] = np.nan
        X_test.loc[mask_test, 'age'] = np.nan

    return X_train, X_test


def label_onehot_encode(X_train, X_test):
    """
    混合编码：
    - 高基数特征：LabelEncoder
    - 低基数特征：OneHotEncoder
    仅用训练集拟合，避免数据泄露
    """
    label_features = ['job', 'education', 'month', 'feature_3', 'feature_4', 'feature_5']
    onehot_features = ['marital', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'poutcome']

    label_features = [f for f in label_features if f in X_train.columns]
    onehot_features = [f for f in onehot_features if f in X_train.columns]

    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    # 1) 标签编码
    for feat in label_features:
        le = LabelEncoder()

        train_vals = X_train_enc[feat].astype(str)
        le.fit(train_vals)
        X_train_enc[feat] = le.transform(train_vals)

        test_vals = X_test_enc[feat].astype(str)
        unseen = set(test_vals) - set(le.classes_)
        if unseen:
            mode_val = train_vals.mode()[0]
            test_vals = test_vals.replace(list(unseen), mode_val)

        X_test_enc[feat] = le.transform(test_vals)

    # 2) 独热编码
    if onehot_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), onehot_features)
            ],
            remainder='passthrough'
        )

        X_train_arr = preprocessor.fit_transform(X_train_enc)
        X_test_arr = preprocessor.transform(X_test_enc)

        onehot_encoder = preprocessor.named_transformers_['onehot']
        onehot_feature_names = onehot_encoder.get_feature_names_out(onehot_features).tolist()
        passthrough_features = [f for f in X_train_enc.columns if f not in onehot_features]
        all_feature_names = onehot_feature_names + passthrough_features

        X_train_enc = pd.DataFrame(X_train_arr, columns=all_feature_names, index=X_train.index)
        X_test_enc = pd.DataFrame(X_test_arr, columns=all_feature_names, index=X_test.index)

    return X_train_enc, X_test_enc


def create_preprocessed_datasets(X_train, X_test, y_train):
    """
    创建两个预处理版本：
    1. median_standard   : 用中位数替换异常值 + StandardScaler
    2. quartile_standard : 用(Q1+Q3)/2替换异常值 + StandardScaler
    """
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

    # 编码目标变量
    le_target = LabelEncoder()
    y_encoded = pd.Series(le_target.fit_transform(y_train), index=y_train.index, name='y')

    datasets = {}

    for version in ['median', 'quartile']:
        X_tr = X_train.copy()
        X_te = X_test.copy()

        for feat in numeric_features:
            Q1 = X_tr[feat].quantile(0.25)
            Q3 = X_tr[feat].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            if version == 'median':
                repl = X_tr[feat].median()
            else:
                repl = (Q1 + Q3) / 2

            X_tr[feat] = np.where((X_tr[feat] < lower) | (X_tr[feat] > upper), repl, X_tr[feat])
            X_te[feat] = np.where((X_te[feat] < lower) | (X_te[feat] > upper), repl, X_te[feat])

        scaler = StandardScaler()
        X_tr[numeric_features] = scaler.fit_transform(X_tr[numeric_features])
        X_te[numeric_features] = scaler.transform(X_te[numeric_features])

        datasets[version] = (X_tr, X_te)

    return (
        datasets['median'][0],
        datasets['median'][1],
        datasets['quartile'][0],
        datasets['quartile'][1],
        y_encoded
    )


def oversample_minority(X_train, y_train, pos_ratio=POS_RATIO):
    """
    过采样正样本至指定比例。
    注意：这个函数仍然保留，但不在预处理阶段调用；
    它会在 train.py 的每个 CV fold 训练子集内被调用。
    """
    df = pd.DataFrame(X_train).copy()
    df['y'] = y_train.values if isinstance(y_train, pd.Series) else y_train

    df_pos = df[df['y'] == 1]
    df_neg = df[df['y'] == 0]

    n_pos_new = int(pos_ratio * len(df_neg))

    # 若本来正样本已经足够多，则不做过采样
    if len(df_pos) >= n_pos_new:
        df_balanced = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=RANDOM_STATE)
    else:
        df_pos_over = df_pos.sample(n=n_pos_new, replace=True, random_state=RANDOM_STATE)
        df_balanced = pd.concat([df_pos_over, df_neg], axis=0).sample(frac=1, random_state=RANDOM_STATE)

    X_bal = df_balanced.drop('y', axis=1)
    y_bal = df_balanced['y']

    return X_bal, y_bal


def run_preprocessing_pipeline():
    """
    完整预处理流程：
    1. 加载原始数据
    2. 基础清洗
    3. 生成 median / quartile 两套版本
    4. 编码
    5. 保存“未过采样”的 processed train/test
    """
    print("=" * 60)
    print("开始数据预处理流水线（不再提前过采样）")
    print("=" * 60)

    # 1. 加载原始数据
    X_train, y_train, X_test = load_raw_data()
    print(f"原始数据形状 - 训练: {X_train.shape}, 测试: {X_test.shape}")

    # 2. 基础清洗
    X_train, X_test = fix_age_outliers(X_train, X_test)
    X_train, X_test = fill_missing_values(X_train, X_test)
    X_train, X_test = handle_infinity_with_quartiles(X_train, X_test)
    X_train, X_test = handle_pdays_special_value(X_train, X_test)

    # 3. 创建两套预处理版本
    train_ms, test_ms, train_qs, test_qs, y_enc = create_preprocessed_datasets(X_train, X_test, y_train)

    # 4. 应用编码
    print("\n应用混合编码...")
    train_ms_enc, test_ms_enc = label_onehot_encode(train_ms, test_ms)
    train_qs_enc, test_qs_enc = label_onehot_encode(train_qs, test_qs)

    # 5. 保存测试数据
    test_ms_enc.to_csv(TEST_MS, index=False)
    test_qs_enc.to_csv(TEST_QS, index=False)
    print(f"测试数据已保存: {TEST_MS}, {TEST_QS}")

    # 6. 保存“未过采样”的训练数据
    pd.concat([train_ms_enc, y_enc.rename('y')], axis=1).to_csv(TRAIN_MS_PROCESSED, index=False)
    pd.concat([train_qs_enc, y_enc.rename('y')], axis=1).to_csv(TRAIN_QS_PROCESSED, index=False)
    print(f"训练数据已保存（未过采样）: {TRAIN_MS_PROCESSED}, {TRAIN_QS_PROCESSED}")

    print("\n预处理流水线完成！")