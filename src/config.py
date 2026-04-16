#src/config.py

#集中管理项目路径、随机种子和常用参数。


import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')

# 原始数据文件
TRAIN_RAW = os.path.join(DATA_DIR, 'bank_marketing_train.csv')
TEST_RAW = os.path.join(DATA_DIR, 'bank_marketing_test.csv')

# 预处理输出文件
TEST_MS = os.path.join(DATA_DIR, 'ms_test_data.csv')
TEST_QS = os.path.join(DATA_DIR, 'qs_test_data.csv')
TRAIN_MS_BALANCED = os.path.join(DATA_DIR, 'train_ms_balanced.csv')
TRAIN_QS_BALANCED = os.path.join(DATA_DIR, 'train_qs_balanced.csv')

# 模型输出
LGB_MODEL_PATH = os.path.join(MODEL_DIR, 'best_lightgbm_model.txt')
ROC_CURVE_PATH = os.path.join(MODEL_DIR, 'lightgbm_roc_curve.png')
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, 'lightgbm_feature_importance.png')

# 随机种子
RANDOM_STATE = 42

# 过采样比例（正样本数 / 负样本数）
POS_RATIO = 0.5

# 交叉验证折数
CV_FOLDS = 5