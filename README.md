# 银行营销预测项目 (Bank Marketing Prediction)

## 项目背景

本项目基于葡萄牙银行机构的直销电话营销数据，旨在预测客户是否会订阅定期存款产品（`y` 变量，二元分类：`yes` / `no`）。数据集来源于 [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)，包含客户基本信息、社会经济指标以及营销活动的历史记录。

在真实的银行营销场景中，识别高潜力客户可以显著提高营销效率、降低沟通成本。因此，构建一个高准确率的预测模型具有重要的商业价值。

## 分析目标

1. **数据清洗与预处理**：处理缺失值、异常值、无穷值，并对分类变量进行合理的编码。
2. **类别不平衡处理**：通过过采样（Oversampling）平衡正负样本比例，避免模型偏向多数类。
3. **模型构建与调优**：使用 LightGBM 作为核心分类器，通过随机搜索（RandomizedSearchCV）结合五折交叉验证寻找最优超参数。
4. **模型评估**：以 AUC（ROC 曲线下面积）为主要评估指标，同时输出特征重要性排名，识别影响客户决策的关键因素。

## 数据处理

### 原始数据

- `bank_marketing_train.csv`：训练集，包含 26,246 条记录，24 个特征 + 1 个目标变量 `y`。
- `bank_marketing_test.csv`：测试集，包含 8,000 条记录，特征与训练集一致，无目标变量。

### 预处理步骤

1. **缺失值处理**：
   - `feature_1`、`feature_2`、`campaign` 在测试集中有少量缺失，分别用中位数、均值和 `0` 填充。
2. **年龄异常值**：
   - 将 `age` 小于 0 或大于 100 的值替换为 `NaN`（实际数据中未出现）。
3. **无穷值处理**：
   - `feature_1` 中存在 `inf`，使用训练集的上、下四分位数进行替换。
4. **特殊值处理**：
   - `pdays` 中的 `999` 表示“从未联系”，统一替换为 `-1` 以保留业务含义。
5. **特征编码**：
   - **标签编码**（Label Encoding）：高基数特征 `job`、`education`、`month`、`feature_3`、`feature_4`、`feature_5`。
   - **独热编码**（One-Hot Encoding）：低基数特征 `marital`、`default`、`housing`、`loan`、`contact`、`day_of_week`、`poutcome`。
6. **数值标准化**：
   - 使用 `StandardScaler` 对数值特征进行标准化（基于训练集拟合）。
7. **异常值处理策略对比**：
   - 生成两个版本的数据集：
     - **median_standard**：基于 IQR 检测异常值，并用**中位数**替换。
     - **quartile_standard**：基于 IQR 检测异常值，并用**上下四分位数均值**替换。
   - 最终建模选用 `median_standard` 版本。
8. **类别不平衡处理**：
   - 原始训练集正负样本比例约为 1:7.4。
   - 采用随机过采样（Random Oversampling）将正样本数量提升至负样本的 **50%**（即 1:2），得到平衡后的训练集 `train_ms_balanced.csv`。

预处理后，特征维度由 24 扩展至 **33** 维。

## 方法流程

### 模型选择

选用 **LightGBM**（Light Gradient Boosting Machine），理由如下：
- 高效处理大规模数据，训练速度快。
- 原生支持类别特征，对内存友好。
- 在结构化数据分类任务中通常表现优异。

### 交叉验证与调优

- **交叉验证策略**：`StratifiedKFold (n_splits=5)`，确保每折中正负样本比例与原始数据一致。
- **超参数搜索**：`RandomizedSearchCV`，从 8748 种组合中随机抽取 20 组进行评估。
- **调优参数空间**：
  - `learning_rate`: [0.01, 0.05, 0.1]
  - `n_estimators`: [100, 150, 200]
  - `num_leaves`: [31, 63, 127]
  - `max_depth`: [6, 8, -1]
  - `min_child_samples`: [20, 50, 100]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.8, 1.0]
  - `reg_alpha`: [0, 0.1, 0.5]
  - `reg_lambda`: [0, 0.1, 0.5]

### 评估指标

- **主要指标**：AUC（Area Under the ROC Curve），衡量模型对正负样本的排序能力。
- **辅助指标**：特征重要性（Gain-based Importance），帮助解释模型决策。

## 结果结论

### 模型性能

| 指标               | 数值    |
|-------------------|---------|
| 交叉验证 AUC (CV)  | 0.9398  |
| 训练集 AUC         | 0.9794  |
| 过拟合程度         | 0.0395  |

模型在交叉验证中取得了 **0.9398** 的 AUC 分数，表明具有优秀的泛化能力。训练集 AUC 与 CV AUC 的差值较小，过拟合控制良好。

### 特征重要性（Top 5）

| 排名 | 特征名称        | 重要性分数 |
|------|----------------|------------|
| 1    | `euribor3m`     | 3525       |
| 2    | `age`           | 2735       |
| 3    | `feature_2`     | 2720       |
| 4    | `feature_5`     | 2639       |
| 5    | `job`           | 1615       |

**解读**：
- **宏观经济指标**（`euribor3m`、`cons.conf.idx`）对客户存款意愿有显著影响。
- **客户年龄**和**职业**是关键的人口统计学特征。
- 匿名特征 `feature_2`、`feature_5` 也提供了较强的预测信号。

### 结论

通过合理的预处理和 LightGBM 模型调优，本项目构建了一个高精度的银行营销响应预测模型。该模型可作为筛选潜在客户的有力工具，帮助业务团队优化营销资源分配。

## 如何运行

### 1. 环境要求

- Python 3.8 及以上版本
- 依赖包见 `requirements.txt`

### 2. 项目结构
```text
bank_marketing_project/
├── data/
│   ├── bank_marketing_train.csv
│   ├── bank_marketing_test.csv
│   ├── train_ms_processed.csv
│   ├── train_qs_processed.csv
│   ├── ms_test_data.csv
│   ├── qs_test_data.csv
│   ├── model_comparison_oof.csv
│   └── test_predictions_best_model.csv
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
├── run_train.py
└── README.md
```
## 3. 环境配置

建议使用 Python 3.8 及以上版本。

安装依赖：

```bash
pip install -r requirements.txt
```
## 4. 运行方式

### Step 1: 数据预处理

预处理模块会完成缺失值处理、异常值处理、无穷值处理、特殊值替换、类别编码、数值标准化，并生成建模所需的 `processed` 数据集。

如果你单独写了预处理入口脚本，可以运行：

```bash
python run_preprocessing.py
```
### Step 2: 模型训练与评估
运行训练脚本：

```bash
python run_train.py
```
训练模块会完成以下任务：

读取 processed training data

使用 5-fold Stratified Cross-Validation 进行模型比较

在每个 fold 的训练子集内部执行 oversampling

基于 OOF（Out-of-Fold）预测计算评估指标

选择最佳模型并生成测试集预测结果




## 5. 输出文件说明

运行完成后，主要输出文件包括：

train_ms_processed.csv：median_standard 版本的训练集（未过采样）

train_qs_processed.csv：quartile_standard 版本的训练集（未过采样）

ms_test_data.csv：median_standard 版本的测试集

qs_test_data.csv：quartile_standard 版本的测试集

model_comparison_oof.csv：多模型横向比较结果（基于 OOF）

test_predictions_best_model.csv：最佳模型在测试集上的预测结果

best_lightgbm_model.txt：保存的 LightGBM 模型文件

*_oof_roc_curve.png：不同模型的 OOF ROC 曲线图

lgb_feature_importance.png：LightGBM 特征重要性图
