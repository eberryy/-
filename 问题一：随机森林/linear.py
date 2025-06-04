import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE

# 使用 matplotlib 内置的中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

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

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=0)

# 特征选择 - 递归特征消除
print("执行特征选择...")
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=10, step=1)
selector = selector.fit(X_train, y_train)

# 获取选择的特征
selected_features = X.columns[selector.support_]
print("选择的特征:", selected_features.tolist())

# 使用选择的特征
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 训练多元线性回归模型
print("训练多元线性回归模型...")
model = LinearRegression()
model.fit(X_train_selected, y_train)

# 预测
y_pred = model.predict(X_test_selected)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n模型评估:")
print(f"均方误差 (MSE): {mse:.2f}")
print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"决定系数 (R²): {r2:.4f}")

# 打印线性回归方程
print("\n线性回归方程:")
print(f"{target_column} = {model.intercept_:.4f}", end="")
for i, feature in enumerate(selected_features):
    coef = model.coef_[i]
    sign = "+" if coef >= 0 else ""
    print(f" {sign} {coef:.6f} × [{feature}]", end="")
print()

# 可视化特征系数
plt.figure(figsize=(12, 8))
coefs = pd.Series(model.coef_, index=selected_features)
coefs_abs = coefs.abs().sort_values(ascending=False)
top_features = coefs_abs.index
coefs = coefs[top_features]
sns.barplot(x=coefs.values, y=top_features)
plt.title('特征系数')
plt.xlabel('系数值')
plt.ylabel('特征')
plt.tight_layout()
plt.savefig('线性回归特征系数.png', dpi=300)
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
plt.savefig('线性回归散点图.png', dpi=300)
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
plt.savefig('线性回归残差图.png', dpi=300)
plt.show()

# 绘制残差分布图
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title('残差分布')
plt.xlabel('残差')
plt.ylabel('频率')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('线性回归残差分布.png', dpi=300)
plt.show()

