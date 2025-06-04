import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import seaborn as sns
import shap
from scipy.stats import randint, uniform
import joblib

# 使用 matplotlib 内置的中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# 定义特征工程函数，用于训练和预测时保持一致的特征处理
def engineer_features(df, is_training=True):
    """
    对数据进行特征工程处理
    
    参数:
    df: 输入的DataFrame
    is_training: 是否为训练模式，如果是则会处理目标变量和日期列
    
    返回:
    处理后的DataFrame，以及目标变量（如果是训练模式）
    """
    # 创建DataFrame的副本，避免修改原始数据
    df_processed = df.copy()
    
    # 查找目标列
    target_column = None
    if is_training:
        target_columns = [col for col in df.columns if "高度" in col or "height" in col.lower()]
        if target_columns:
            target_column = target_columns[0]
            print(f"找到目标列: {target_column}")
        else:
            target_column = df.columns[-1]
            print(f"未找到明确的目标列，使用最后一列: {target_column}")
    
    # 处理日期列
    if "date" in df_processed.columns and is_training:
        df_processed["date"] = pd.to_datetime(df_processed["date"])
        df_processed.sort_values("date", inplace=True)
    
    # 提取时间特征
    if "date" in df_processed.columns:
        df_processed["Hour"] = pd.to_datetime(df_processed["date"]).dt.hour
        df_processed["Month"] = pd.to_datetime(df_processed["date"]).dt.month
        df_processed["Season"] = pd.to_datetime(df_processed["date"]).dt.month.map(
            {1:0, 2:0, 3:1, 4:1, 5:1, 6:2, 7:2, 8:2, 9:3, 10:3, 11:3, 12:0}
        )
    
    # 计算风速
    if "U向分量风/m/s" in df_processed.columns and "V向分量风/m/s" in df_processed.columns:
        df_processed["WS"] = np.sqrt(df_processed["U向分量风/m/s"]**2 + df_processed["V向分量风/m/s"]**2)
        
        # 剔除异常值（仅在训练模式下）
        if is_training:
            df_processed.loc[df_processed["WS"] > 30, "WS"] = np.nan
    
    # 处理温度异常值（仅在训练模式下）
    if "2m气温/K" in df_processed.columns and is_training:
        df_processed.loc[df_processed["2m气温/K"] < 250, "2m气温/K"] = np.nan
        df_processed.loc[df_processed["2m气温/K"] > 315, "2m气温/K"] = np.nan
    
    # 计算温度梯度
    if "地表温度/K" in df_processed.columns and "2m气温/K" in df_processed.columns:
        df_processed["ΔT"] = df_processed["地表温度/K"] - df_processed["2m气温/K"]
        
        # 计算风速与温度梯度的交互项
        if "WS" in df_processed.columns:
            df_processed["WSxΔT"] = df_processed["WS"] * df_processed["ΔT"]
    
    # 计算湿度与气压的交互项
    if "相对湿度/%" in df_processed.columns and "表面气压/Pa" in df_processed.columns:
        df_processed["RHxP"] = df_processed["相对湿度/%"] * df_processed["表面气压/Pa"]

    # 添加更多的特征交互项
    if "WS" in df_processed.columns and "相对湿度/%" in df_processed.columns:
        df_processed["WSxRH"] = df_processed["WS"] * df_processed["相对湿度/%"]
    
    if "2m气温/K" in df_processed.columns and "相对湿度/%" in df_processed.columns:
        df_processed["TxRH"] = df_processed["2m气温/K"] * df_processed["相对湿度/%"]
    
    # 添加二次项特征
    if "WS" in df_processed.columns:
        df_processed["WS_squared"] = df_processed["WS"] ** 2
    
    if "ΔT" in df_processed.columns:
        df_processed["ΔT_squared"] = df_processed["ΔT"] ** 2
    
    # 添加时间的周期性特征
    if "Hour" in df_processed.columns:
        df_processed["Hour_sin"] = np.sin(2 * np.pi * df_processed["Hour"] / 24)
        df_processed["Hour_cos"] = np.cos(2 * np.pi * df_processed["Hour"] / 24)
    
    if "Month" in df_processed.columns:
        df_processed["Month_sin"] = np.sin(2 * np.pi * df_processed["Month"] / 12)
        df_processed["Month_cos"] = np.cos(2 * np.pi * df_processed["Month"] / 12)
    
    # 线性插值填充缺失值（仅在训练模式下）
    if is_training:
        df_processed.interpolate(method="linear", inplace=True)
    
    # 准备特征和目标变量
    if is_training and target_column:
        # 移除不需要的列
        drop_columns = ["date"] if "date" in df_processed.columns else []
        drop_columns.append(target_column)
        X = df_processed.drop(drop_columns, axis=1)
        y = df_processed[target_column]
        return X, y, list(X.columns)
    else:
        # 预测模式，只返回特征
        if "date" in df_processed.columns:
            df_processed = df_processed.drop(["date"], axis=1)
        return df_processed, None, list(df_processed.columns)

# 加载数据
path = "c:\\Users\\eberry\\Desktop\\python\\mathmodelling\\ABLHdata.xlsx"
df = pd.read_excel(path)

# 检查数据列
print("数据列名:", df.columns.tolist())
print("数据形状:", df.shape)

# 应用特征工程
X, y, feature_columns = engineer_features(df, is_training=True)

# 数据预览
print("处理后数据预览:")
print(X.head())
print("特征列:", feature_columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义XGBoost参数分布
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'min_child_weight': randint(1, 10),
    'gamma': uniform(0, 1),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1)
}

# 创建XGBoost回归器
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# 创建随机搜索对象
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50,  # 尝试50种不同的组合
    cv=5,
    n_jobs=-1,
    scoring='r2',
    verbose=1,
    random_state=42
)

# 扩展超参数搜索空间
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.005, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'min_child_weight': randint(1, 20),
    'gamma': uniform(0, 2),
    'reg_alpha': uniform(0, 10),
    'reg_lambda': uniform(0, 10),
    'scale_pos_weight': uniform(0.5, 2)
}

# 增加搜索迭代次数
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=100,  # 增加到100次迭代
    cv=5,
    n_jobs=-1,
    scoring='neg_mean_squared_error',  # 使用MSE而不是R2
    verbose=1,
    random_state=42
)

# 执行随机搜索
print("开始随机搜索最佳参数...")
random_search.fit(X_train, y_train)
print("随机搜索完成。")

# 输出最佳参数
print("最佳参数:", random_search.best_params_)
print("最佳交叉验证得分:", random_search.best_score_)

# 使用最佳参数创建模型
best_xgb = random_search.best_estimator_

# 在测试集上评估
y_pred = best_xgb.predict(X_test)

# 评估回归性能
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型评估:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"决定系数 (R²): {r2:.4f}")

# 获取特征重要性
importance = best_xgb.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("\n特征重要性:")
print(feature_importance)

# 可视化特征重要性
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('XGBoost特征重要性')
plt.tight_layout()
plt.savefig('XGBoost特征重要性.png', dpi=300)
plt.show()

# 绘制预测值与实际值的散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title(f'真实值 vs. 预测值\nR$^{2}$ = {r2:.4f}')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('XGBoost散点图.png', dpi=300)
plt.show()

# 绘制残差图
plt.figure(figsize=(10, 6))
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('残差图')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('XGBoost残差图.png', dpi=300)
plt.show()

# 绘制残差分布图
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('残差分布')
plt.xlabel('残差')
plt.ylabel('频率')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('XGBoost残差分布.png', dpi=300)
plt.show()

# SHAP值分析
print("计算SHAP值...")
# 创建SHAP解释器
explainer = shap.Explainer(best_xgb)

# 计算SHAP值
# 使用一个小的样本来加速计算（如果数据集很大）
X_sample = X_test.iloc[:100] if len(X_test) > 100 else X_test
shap_values = explainer(X_sample)

# SHAP摘要图 - 显示所有特征的整体影响
plt.figure(figsize=(12, 8))
shap.plots.bar(shap_values)
plt.title('SHAP特征重要性摘要')
plt.tight_layout()
plt.savefig('XGBoost_SHAP_bar.png', dpi=300)
plt.show()

# SHAP摘要图 - 显示每个特征的分布和影响
plt.figure(figsize=(12, 10))
shap.plots.beeswarm(shap_values)
plt.title('SHAP特征影响分布')
plt.tight_layout()
plt.savefig('XGBoost_SHAP_beeswarm.png', dpi=300)
plt.show()

# 保存模型
best_xgb.save_model('best_xgboost_model.json')

# 保存特征列名，用于预测时确保特征顺序一致
feature_info = {
    'feature_columns': feature_columns
}
joblib.dump(feature_info, 'feature_info.joblib')

print("XGBoost分析完成。")
print("特征列已保存，可用于预测时确保特征顺序一致。")
