import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import randint, uniform
import shap  

# 使用 matplotlib 内置的中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']  # 添加DejaVu Sans作为备选
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体集处理数学符号

# 加载数据
path = "D:\\en\\ABLHdata.xlsx"
df = pd.read_excel(path)

# 检查数据列
print("数据列名:", df.columns.tolist())
print("数据形状:", df.shape)

# 检查目标列
target_column = "大气边界层高度/m"

# 检查是否存在目标列
if target_column not in df.columns:
    print(f"警告: 未找到目标列 '{target_column}'，请检查数据或修改目标列名")
    # 尝试查找可能的目标列
    possible_targets = [col for col in df.columns if "高度" in col or "height" in col.lower()]
    if possible_targets:
        print(f"可能的目标列: {possible_targets}")
        target_column = possible_targets[0]
        print(f"使用 '{target_column}' 作为目标列")
    else:
        # 如果找不到可能的目标列，使用最后一列作为目标
        target_column = df.columns[-1]
        print(f"未找到可能的目标列，使用最后一列 '{target_column}' 作为目标")

# 处理日期列
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    
    # 时间特征
    df["Hour"] = df["date"].dt.hour
    df["Month"] = df["date"].dt.month
    df["Season"] = df["date"].dt.month.map({1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0})

# 检查是否存在风速分量列
if "U向分量风/m/s" in df.columns and "V向分量风/m/s" in df.columns:
    # 合成风速
    df["WS"] = np.sqrt(df["U向分量风/m/s"]**2 + df["V向分量风/m/s"]**2)
    
    # 剔除异常值
    df.loc[df["WS"] > 30, "WS"] = np.nan

# 检查是否存在温度列
if "2m气温/K" in df.columns:
    # 剔除异常值
    df.loc[df["2m气温/K"] < 250, "2m气温/K"] = np.nan
    df.loc[df["2m气温/K"] > 315, "2m气温/K"] = np.nan

# 检查是否存在地表温度列
if "地表温度/K" in df.columns and "2m气温/K" in df.columns:
    # 温度梯度 ΔT
    df["ΔT"] = df["地表温度/K"] - df["2m气温/K"]
    
    # 检查是否存在风速列
    if "WS" in df.columns:
        # 交互项
        df["WSxΔT"] = df["WS"] * df["ΔT"]

# 检查是否存在相对湿度和气压列
if "相对湿度/%" in df.columns and "表面气压/Pa" in df.columns:
    # 交互项
    df["RHxP"] = df["相对湿度/%"] * df["表面气压/Pa"]

# 线性插值填充缺失值
df.interpolate(method="linear", inplace=True)

# 数据预览
print("处理后数据预览:")
print(df.head())

# 准备特征和目标变量
# 移除不需要的列
drop_columns = ["date"] if "date" in df.columns else []
drop_columns.append(target_column)
X = df.drop(drop_columns, axis=1)
y = df[target_column]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 定义参数分布
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(10, 50),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2', None],
    'bootstrap': [True, False]
}

# 创建随机搜索对象
random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=0),
    param_distributions=param_dist,
    n_iter=100,  # 尝试100种不同的组合
    cv=5,
    n_jobs=-1,
    scoring='r2',
    verbose=1,
    random_state=0
)

# 执行随机搜索
print("开始随机搜索...")
random_search.fit(X_train, y_train)
print("随机搜索完成。")

# 输出最佳参数
print("最佳参数:", random_search.best_params_)
print("最佳交叉验证得分:", random_search.best_score_)

# 使用最佳参数创建模型
best_rf = random_search.best_estimator_

# 在测试集上评估
y_pred = best_rf.predict(X_test)

# 评估回归性能
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, y_pred)))

# 计算特征重要性
importances = best_rf.feature_importances_
print("特征重要性:")
for i, feature_name in enumerate(X.columns):
    print(f"{feature_name}: {importances[i]:.4f}")

# 计算R²
r2 = r2_score(y_test, y_pred)
print("R²:", r2)

# 绘制测试集散点图和斜线
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='grey', linewidth=2)
plt.title(f'真实值 vs. 预测值\nR$^{2}$ = {r2:.2f}')
plt.xlabel('真实值 (y_test)')
plt.ylabel('预测值 (y_pred)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('散点图.png', dpi=300)
plt.show()

# 可视化特征重要性
feature_names = X.columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12, 8))
plt.title('特征重要性')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('特征重要性.png', dpi=300)
plt.show()

# ====================== SHAP 值计算 ======================
print("计算SHAP值...")

# 创建SHAP解释器
# 使用一个小的样本来加速计算（如果数据集很大）  
X_sample = X_test.iloc[:100] if len(X_test) > 100 else X_test

# 创建SHAP解释器 - 使用新的方式
explainer = shap.Explainer(best_rf)

# 计算SHAP值 - 这将返回一个Explanation对象
shap_values = explainer(X_sample)

# 1. SHAP摘要图 - 显示所有特征的整体影响
plt.figure(figsize=(12, 8))
shap.plots.bar(shap_values)
ax = plt.gca()
ax.set_title('SHAP特征重要性摘要')
plt.tight_layout()
plt.show()

# 2. SHAP摘要图 - 显示每个特征的分布和影响
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_values)
ax = plt.gca()
ax.set_title('SHAP特征影响分布')
plt.tight_layout()
plt.show()

print("SHAP分析完成。")


# 获取基准值（平均预测值）
base_value = explainer.expected_value
# 如果base_value是数组，取第一个元素
if isinstance(base_value, np.ndarray):
    base_value = base_value[0]

# 计算每个特征的平均绝对SHAP值
mean_abs_shap = np.abs(shap_values.values).mean(0)
# 按SHAP值大小排序特征
shap_importance = np.argsort(mean_abs_shap)[::-1]

# 计算每个特征的平均SHAP值（考虑正负）
mean_shap = shap_values.values.mean(0)

# 获取特征的平均值，用于标准化
feature_means = X_sample.mean(0)

print("特征贡献 (按重要性排序):")
for i in range(15):
    idx = shap_importance[i]
    feature_name = X.columns[idx]
    abs_shap = mean_abs_shap[idx]
    shap_val = mean_shap[idx]
    direction = "正向" if shap_val > 0 else "负向"
    print(f"{i+1}. {feature_name}: 平均贡献 = {shap_val:.4f}, 重要性 = {abs_shap:.4f}, 影响方向: {direction}")






