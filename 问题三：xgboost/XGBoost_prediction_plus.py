import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, request, render_template, jsonify, send_file
import os
import json
import traceback
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# 设置matplotlib的中文和数学符号支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']  # 添加DejaVu Sans作为备选
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用Computer Modern字体集处理数学符号

app = Flask(__name__)

# 尝试多个可能的模型路径
possible_model_paths = [
    'best_xgboost_model.json',  # 当前目录
    'C:\\Users\\eberry\\Desktop\\python\\best_xgboost_model.json',  # 原始路径
    'C:\\Users\\eberry\\Desktop\\python\\mathmodelling\\best_xgboost_model.json'  # mathmodelling目录
]

MODEL_PATH = None
for path in possible_model_paths:
    if os.path.exists(path):
        MODEL_PATH = path
        print(f"找到模型文件: {MODEL_PATH}")
        break

if MODEL_PATH is None:
    print("警告: 未找到模型文件，请确保已运行XGBoost.py训练模型")
    print("将尝试继续运行，但预测功能将不可用")
    print(f"尝试过的路径: {possible_model_paths}")

# 尝试加载模型
model = None
try:
    if MODEL_PATH:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_PATH)
        print("模型加载成功")
except Exception as e:
    print(f"加载模型时出错: {str(e)}")
    traceback.print_exc()

# 尝试加载特征信息
feature_info = None
feature_columns = []
try:
    feature_info_path = 'feature_info.joblib'
    if os.path.exists(feature_info_path):
        feature_info = joblib.load(feature_info_path)
        feature_columns = feature_info.get('feature_columns', [])
        print(f"加载了{len(feature_columns)}个特征列")
    else:
        print("未找到特征信息文件，将使用默认特征列")
except Exception as e:
    print(f"加载特征信息时出错: {str(e)}")
    traceback.print_exc()

# 定义特征工程函数，与XGBoost.py保持一致
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

# 获取原始特征列（用于前端表单）
def get_raw_features():
    # 尝试从ABLHdata.xlsx获取原始特征
    try:
        path = "ABLHdata.xlsx"
        if not os.path.exists(path):
            path = "C:\\Users\\eberry\\Desktop\\python\\mathmodelling\\ABLHdata.xlsx"
        
        if os.path.exists(path):
            df = pd.read_excel(path)
            # 排除日期列和目标列
            raw_features = [col for col in df.columns if col != "date" and "高度" not in col.lower() and "height" not in col.lower()]
            return raw_features
        else:
            print(f"未找到数据文件: {path}")
            return []
    except Exception as e:
        print(f"获取原始特征时出错: {str(e)}")
        traceback.print_exc()
        return []

# 获取原始特征列
raw_features = get_raw_features()
print(f"原始特征列: {raw_features}")

# 确保templates目录存在
try:
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    print(f"模板目录: {templates_dir}")
    
    # 创建HTML模板
    template_path = os.path.join(templates_dir, 'index.html')
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>大气边界层高度(ABLH)预测</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: inline-block;
            width: 200px;
            text-align: right;
            margin-right: 10px;
        }
        input[type="number"], input[type="datetime-local"] {
            width: 150px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            display: block;
            margin: 20px auto;
        }
        button:hover {
            background-color: #2980b9;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            text-align: center;
            font-size: 18px;
            display: none;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 10px;
            display: none;
        }
        .status {
            background-color: #f1c40f;
            color: #333;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 4px;
            text-align: center;
        }
        .section {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .section h2 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 18px;
        }
        .calculated-features {
            background-color: #eaf2f8;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }
        .calculated-features h3 {
            margin-top: 0;
            color: #2980b9;
            font-size: 16px;
        }
        .feature-value {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
        }
        .feature-name {
            font-weight: bold;
        }
        .tabs {
            display: flex;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            background-color: #f1f1f1;
            border: 1px solid #ddd;
            cursor: pointer;
            margin-right: 5px;
            border-radius: 4px 4px 0 0;
        }
        .tab.active {
            background-color: #3498db;
            color: white;
            border-color: #3498db;
        }
        .tab-content {
            display: none;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 0 4px 4px 4px;
        }
        .tab-content.active {
            display: block;
        }
        .nav-link {
            display: inline-block;
            margin: 10px;
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .nav-link:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>大气边界层高度(ABLH)预测</h1>
    
    <div id="status-message" class="status" style="display: none;"></div>
    
    <div style="text-align: center; margin-bottom: 20px;">
        <a href="/" class="nav-link">单条预测</a>
        <a href="/batch_test" class="nav-link">批量预测</a>
    </div>
    
    <form id="prediction-form">
        <div class="section">
            <h2>时间信息</h2>
            <div class="form-group">
                <label for="datetime">日期时间:</label>
                <input type="datetime-local" id="datetime" name="datetime" required>
            </div>
        </div>
        
        <div class="section">
            <h2>原始气象数据</h2>
            <div id="raw-input-fields">
                <!-- 原始输入字段将通过JavaScript动态生成 -->
            </div>
        </div>
        
        <div class="calculated-features">
            <h3>计算的特征值</h3>
            <div id="calculated-features">
                <!-- 计算的特征将在这里显示 -->
            </div>
        </div>
        
        <button type="submit">预测ABLH</button>
    </form>
    <div id="error" class="error"></div>
    <div id="result" class="result"></div>

    <script>
        // 特征列表
        const features = {{features|tojson}};
        const rawFeatures = {{raw_features|tojson}};
        
        // 检查模型状态
        fetch('/model_status')
            .then(response => response.json())
            .then(data => {
                if (!data.model_loaded) {
                    document.getElementById('status-message').textContent = data.message;
                    document.getElementById('status-message').style.display = 'block';
                    document.getElementById('prediction-form').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('获取模型状态时出错:', error);
            });
        
        // 动态生成原始输入字段
        const rawInputFields = document.getElementById('raw-input-fields');
        
        if (rawFeatures && rawFeatures.length > 0) {
            rawFeatures.forEach(feature => {
                const formGroup = document.createElement('div');
                formGroup.className = 'form-group';
                
                const label = document.createElement('label');
                label.textContent = feature + ':';
                label.setAttribute('for', feature);
                
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.id = feature;
                input.name = feature;
                input.required = true;
                
                // 添加输入事件监听器，当输入改变时重新计算特征
                input.addEventListener('input', calculateFeatures);
                
                formGroup.appendChild(label);
                formGroup.appendChild(input);
                rawInputFields.appendChild(formGroup);
            });
        } else {
            rawInputFields.innerHTML = '<p>无法加载原始特征列，请检查服务器日志。</p>';
        }
        
        // 日期时间输入框添加事件监听器
        document.getElementById('datetime').addEventListener('input', calculateFeatures);
        
        // 在界面上显示特征值
        function addFeatureDisplay(name, value) {
            const div = document.createElement('div');
            div.className = 'feature-value';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'feature-name';
            nameSpan.textContent = name + ':';
            
            const valueSpan = document.createElement('span');
            valueSpan.textContent = value;
            
            div.appendChild(nameSpan);
            div.appendChild(valueSpan);
            return div;
        }
        
        // 计算特征函数
        function calculateFeatures() {
            const datetime = document.getElementById('datetime').value;
            if (!datetime) return;
            
            // 解析日期时间
            const date = new Date(datetime);
            const hour = date.getHours();
            const month = date.getMonth() + 1; // 月份从0开始，所以+1
            
            // 计算季节 (1-3冬季=0, 4-6春季=1, 7-9夏季=2, 10-12秋季=3)
            let season;
            if (month >= 1 && month <= 3) season = 0;
            else if (month >= 4 && month <= 6) season = 1;
            else if (month >= 7 && month <= 9) season = 2;
            else season = 3;
            
            // 获取其他输入值
            let uWind = 0, vWind = 0, surfaceTemp = 0, airTemp = 0, humidity = 0, pressure = 0;
            
            // 尝试获取风速分量
            if (document.getElementById('U向分量风/m/s')) {
                uWind = parseFloat(document.getElementById('U向分量风/m/s').value) || 0;
            }
            if (document.getElementById('V向分量风/m/s')) {
                vWind = parseFloat(document.getElementById('V向分量风/m/s').value) || 0;
            }
            
            // 计算风速
            const ws = Math.sqrt(uWind * uWind + vWind * vWind);
            
            // 尝试获取温度
            if (document.getElementById('地表温度/K')) {
                surfaceTemp = parseFloat(document.getElementById('地表温度/K').value) || 0;
            }
            if (document.getElementById('2m气温/K')) {
                airTemp = parseFloat(document.getElementById('2m气温/K').value) || 0;
            }
            
            // 计算温度梯度
            const deltaT = surfaceTemp - airTemp;
            
            // 计算风速与温度梯度的交互项
            const wsXdeltaT = ws * deltaT;
            
            // 尝试获取湿度和气压
            if (document.getElementById('相对湿度/%')) {
                humidity = parseFloat(document.getElementById('相对湿度/%').value) || 0;
            }
            if (document.getElementById('表面气压/Pa')) {
                pressure = parseFloat(document.getElementById('表面气压/Pa').value) || 0;
            }
            
            // 计算湿度与气压的交互项
            const rhXp = humidity * pressure;
            
            // 添加更多的特征交互项
            const wsXrh = ws * humidity;
            const tXrh = airTemp * humidity;
            
            // 添加二次项特征
            const ws_squared = ws * ws;
            const deltaT_squared = deltaT * deltaT;
            
            // 添加时间的周期性特征
            const hour_sin = Math.sin(2 * Math.PI * hour / 24);
            const hour_cos = Math.cos(2 * Math.PI * hour / 24);
            const month_sin = Math.sin(2 * Math.PI * month / 12);
            const month_cos = Math.cos(2 * Math.PI * month / 12);
            
            // 存储计算的特征值到全局变量
            calculatedFeatures = {
                'Hour': hour,
                'Month': month,
                'Season': season,
                'WS': ws,
                'ΔT': deltaT,
                'WSxΔT': wsXdeltaT,
                'RHxP': rhXp,
                'WSxRH': wsXrh,
                'TxRH': tXrh,
                'WS_squared': ws_squared,
                'ΔT_squared': deltaT_squared,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Month_sin': month_sin,
                'Month_cos': month_cos
            };
            
            // 显示计算的特征
            const calculatedFeaturesDiv = document.getElementById('calculated-features');
            calculatedFeaturesDiv.innerHTML = '';
            
            // 添加计算的特征
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('小时(Hour)', hour));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('月份(Month)', month));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('季节(Season)', season));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速(WS)', ws.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('温度梯度(ΔT)', deltaT.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速x温度梯度(WSxΔT)', wsXdeltaT.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('湿度x气压(RHxP)', rhXp.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速x湿度(WSxRH)', wsXrh.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('温度x湿度(TxRH)', tXrh.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速平方(WS_squared)', ws_squared.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('温度梯度平方(ΔT_squared)', deltaT_squared.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('小时正弦(Hour_sin)', hour_sin.toFixed(4)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('小时余弦(Hour_cos)', hour_cos.toFixed(4)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('月份正弦(Month_sin)', month_sin.toFixed(4)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('月份余弦(Month_cos)', month_cos.toFixed(4)));
        }
        
        // 表单提交处理
        document.getElementById('prediction-form').addEventListener('submit', function(event) {
            event.preventDefault();
            
            // 获取所有输入值
            const formData = {
                datetime: document.getElementById('datetime').value,
                raw_features: {},
                calculated_features: calculatedFeatures
            };
            
            // 获取所有原始特征的值
            if (rawFeatures && Array.isArray(rawFeatures)) {
                rawFeatures.forEach(feature => {
                    const input = document.getElementById(feature);
                    if (input) {
                        formData.raw_features[feature] = parseFloat(input.value) || 0;
                    }
                });
            }
            
            // 发送预测请求
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                const resultDiv = document.getElementById('result');
                const errorDiv = document.getElementById('error');
                
                if (data.error) {
                    errorDiv.textContent = data.error;
                    errorDiv.style.display = 'block';
                    resultDiv.style.display = 'none';
                } else {
                    resultDiv.textContent = `预测的大气边界层高度(ABLH)为: ${data.prediction.toFixed(2)} 米`;
                    resultDiv.style.display = 'block';
                    errorDiv.style.display = 'none';
                }
            })
            .catch(error => {
                console.error('预测请求出错:', error);
                const errorDiv = document.getElementById('error');
                errorDiv.textContent = '预测请求失败，请检查网络连接或联系管理员。';
                errorDiv.style.display = 'block';
                document.getElementById('result').style.display = 'none';
            });
        });
        
        // 初始化时计算一次特征
        document.addEventListener('DOMContentLoaded', function() {
            // 设置默认日期时间为当前时间
            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            const hour = String(now.getHours()).padStart(2, '0');
            const minute = String(now.getMinutes()).padStart(2, '0');
            
            document.getElementById('datetime').value = `${year}-${month}-${day}T${hour}:${minute}`;
            
            // 初始计算特征
            calculateFeatures();
        });
    </script>
</body>
</html>
        ''')
    print(f"已创建HTML模板: {template_path}")
    
    # 创建批量测试页面
    batch_template_path = os.path.join(templates_dir, 'batch_test.html')
    with open(batch_template_path, 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>ABLH批量预测与可视化</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #2c3e50;
            text-align: center;
        }
        .section {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            display: block;
            margin: 20px auto;
        }
        button:hover {
            background-color: #2980b9;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 10px;
            display: none;
        }
        .result-container {
            margin-top: 20px;
            display: none;
        }
        .metrics {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            margin: 10px;
            text-align: center;
            min-width: 150px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }
        .chart-container {
            width: 100%;
            margin-bottom: 30px;
            text-decoration: none;
            border-radius: 4px;
        }
        .nav-link:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>大气边界层高度(ABLH)预测</h1>
    
    <div id="status-message" class="status" style="display: none;"></div>
    
    <div style="text-align: center; margin-bottom: 20px;">
        <a href="/" class="nav-link">单条预测</a>
        <a href="/batch_test" class="nav-link">批量预测</a>
    </div>
    
    <form id="prediction-form">
        <div class="section">
            <h2>时间信息</h2>
            <div class="form-group">
                <label for="datetime">日期时间:</label>
                <input type="datetime-local" id="datetime" name="datetime" required>
            </div>
        </div>
        
        <div class="section">
            <h2>原始气象数据</h2>
            <div id="raw-input-fields">
                <!-- 原始输入字段将通过JavaScript动态生成 -->
            </div>
        </div>
        
        <div class="calculated-features">
            <h3>计算的特征值</h3>
            <div id="calculated-features">
                <!-- 计算的特征将在这里显示 -->
            </div>
        </div>
        
        <button type="submit">预测ABLH</button>
    </form>
    <div id="error" class="error"></div>
    <div id="result" class="result"></div>

    <script>
        // 特征列表
        const features = {{features|tojson}};
        const rawFeatures = {{raw_features|tojson}};
        
        // 检查模型状态
        fetch('/model_status')
            .then(response => response.json())
            .then(data => {
                if (!data.model_loaded) {
                    document.getElementById('status-message').textContent = data.message;
                    document.getElementById('status-message').style.display = 'block';
                    document.getElementById('prediction-form').style.display = 'none';
                }
            })
            .catch(error => {
                console.error('获取模型状态时出错:', error);
            });
        
        // 动态生成原始输入字段
        const rawInputFields = document.getElementById('raw-input-fields');
        
        if (rawFeatures && rawFeatures.length > 0) {
            rawFeatures.forEach(feature => {
                const formGroup = document.createElement('div');
                formGroup.className = 'form-group';
                
                const label = document.createElement('label');
                label.textContent = feature + ':';
                label.setAttribute('for', feature);
                
                const input = document.createElement('input');
                input.type = 'number';
                input.step = 'any';
                input.id = feature;
                input.name = feature;
                input.required = true;
                
                // 添加输入事件监听器，当输入改变时重新计算特征
                input.addEventListener('input', calculateFeatures);
                
                formGroup.appendChild(label);
                formGroup.appendChild(input);
                rawInputFields.appendChild(formGroup);
            });
        } else {
            rawInputFields.innerHTML = '<p>无法加载原始特征列，请检查服务器日志。</p>';
        }
        
        // 日期时间输入框添加事件监听器
        document.getElementById('datetime').addEventListener('input', calculateFeatures);
        
        // 在界面上显示特征值
        function addFeatureDisplay(name, value) {
            const div = document.createElement('div');
            div.className = 'feature-value';
            
            const nameSpan = document.createElement('span');
            nameSpan.className = 'feature-name';
            nameSpan.textContent = name + ':';
            
            const valueSpan = document.createElement('span');
            valueSpan.textContent = value;
            
            div.appendChild(nameSpan);
            div.appendChild(valueSpan);
            return div;
        }
        
        // 计算特征函数
        function calculateFeatures() {
            const datetime = document.getElementById('datetime').value;
            if (!datetime) return;
            
            // 解析日期时间
            const date = new Date(datetime);
            const hour = date.getHours();
            const month = date.getMonth() + 1; // 月份从0开始，所以+1
            
            // 计算季节 (1-3冬季=0, 4-6春季=1, 7-9夏季=2, 10-12秋季=3)
            let season;
            if (month >= 1 && month <= 3) season = 0;
            else if (month >= 4 && month <= 6) season = 1;
            else if (month >= 7 && month <= 9) season = 2;
            else season = 3;
            
            // 获取其他输入值
            let uWind = 0, vWind = 0, surfaceTemp = 0, airTemp = 0, humidity = 0, pressure = 0;
            
            // 尝试获取风速分量
            if (document.getElementById('U向分量风/m/s')) {
                uWind = parseFloat(document.getElementById('U向分量风/m/s').value) || 0;
            }
            if (document.getElementById('V向分量风/m/s')) {
                vWind = parseFloat(document.getElementById('V向分量风/m/s').value) || 0;
            }
            
            // 计算风速
            const ws = Math.sqrt(uWind * uWind + vWind * vWind);
            
            // 尝试获取温度
            if (document.getElementById('地表温度/K')) {
                surfaceTemp = parseFloat(document.getElementById('地表温度/K').value) || 0;
            }
            if (document.getElementById('2m气温/K')) {
                airTemp = parseFloat(document.getElementById('2m气温/K').value) || 0;
            }
            
            // 计算温度梯度
            const deltaT = surfaceTemp - airTemp;
            
            // 计算风速与温度梯度的交互项
            const wsXdeltaT = ws * deltaT;
            
            // 尝试获取湿度和气压
            if (document.getElementById('相对湿度/%')) {
                humidity = parseFloat(document.getElementById('相对湿度/%').value) || 0;
            }
            if (document.getElementById('表面气压/Pa')) {
                pressure = parseFloat(document.getElementById('表面气压/Pa').value) || 0;
            }
            
            // 计算湿度与气压的交互项
            const rhXp = humidity * pressure;
            
            // 添加更多的特征交互项
            const wsXrh = ws * humidity;
            const tXrh = airTemp * humidity;
            
            // 添加二次项特征
            const ws_squared = ws * ws;
            const deltaT_squared = deltaT * deltaT;
            
            // 添加时间的周期性特征
            const hour_sin = Math.sin(2 * Math.PI * hour / 24);
            const hour_cos = Math.cos(2 * Math.PI * hour / 24);
            const month_sin = Math.sin(2 * Math.PI * month / 12);
            const month_cos = Math.cos(2 * Math.PI * month / 12);
            
            // 存储计算的特征值到全局变量
            calculatedFeatures = {
                'Hour': hour,
                'Month': month,
                'Season': season,
                'WS': ws,
                'ΔT': deltaT,
                'WSxΔT': wsXdeltaT,
                'RHxP': rhXp,
                'WSxRH': wsXrh,
                'TxRH': tXrh,
                'WS_squared': ws_squared,
                'ΔT_squared': deltaT_squared,
                'Hour_sin': hour_sin,
                'Hour_cos': hour_cos,
                'Month_sin': month_sin,
                'Month_cos': month_cos
            };
            
            // 显示计算的特征
            const calculatedFeaturesDiv = document.getElementById('calculated-features');
            calculatedFeaturesDiv.innerHTML = '';
            
            // 添加计算的特征
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('小时(Hour)', hour));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('月份(Month)', month));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('季节(Season)', season));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速(WS)', ws.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('温度梯度(ΔT)', deltaT.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速x温度梯度(WSxΔT)', wsXdeltaT.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('湿度x气压(RHxP)', rhXp.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速x湿度(WSxRH)', wsXrh.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('温度x湿度(TxRH)', tXrh.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('风速平方(WS_squared)', ws_squared.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('温度梯度平方(ΔT_squared)', deltaT_squared.toFixed(2)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('小时正弦(Hour_sin)', hour_sin.toFixed(4)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('小时余弦(Hour_cos)', hour_cos.toFixed(4)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('月份正弦(Month_sin)', month_sin.toFixed(4)));
            calculatedFeaturesDiv.appendChild(addFeatureDisplay('月份余弦(Month_cos)', month_cos.toFixed(4)));
        }
        
        // 表单提交处理
        document.getElementById('prediction-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            // 获取日期时间
            const datetime = document.getElementById('datetime').value;
            if (!datetime) {
                document.getElementById('error').textContent = '请选择日期时间';
                document.getElementById('error').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                return;
            }
            
            // 构建预测数据
            const formData = {
                ...calculatedFeatures
            };
            
            // 添加原始特征
            rawFeatures.forEach(feature => {
                const input = document.getElementById(feature);
                if (input) {
                    formData[feature] = parseFloat(input.value) || 0;
                }
            });
            
            try {
                // 发送预测请求
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData),
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('result').style.display = 'none';
                } else {
                    document.getElementById('result').textContent = `预测的大气边界层高度(ABLH): ${data.prediction.toFixed(2)} 米`;
                    document.getElementById('result').style.display = 'block';
                    document.getElementById('error').style.display = 'none';
                }
            } catch (error) {
                document.getElementById('error').textContent = '预测过程中发生错误，请稍后再试';
                document.getElementById('error').style.display = 'block';
                document.getElementById('result').style.display = 'none';
            }
        });
        
        // 初始化时计算一次特征
        document.addEventListener('DOMContentLoaded', function() {
            // 设置默认日期时间为当前时间
            const now = new Date();
            const year = now.getFullYear();
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const day = String(now.getDate()).padStart(2, '0');
            const hour = String(now.getHours()).padStart(2, '0');
            const minute = String(now.getMinutes()).padStart(2, '0');
            
            document.getElementById('datetime').value = `${year}-${month}-${day}T${hour}:${minute}`;
            
            // 初始计算特征
            calculateFeatures();
        });
        
        // 切换标签页函数
        function openTab(evt, tabName) {
            // 隐藏所有标签页内容
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].style.display = 'none';
            }
            
            // 移除所有标签的active类
            const tabs = document.getElementsByClassName('tab');
            for (let i = 0; i < tabs.length; i++) {
                tabs[i].className = tabs[i].className.replace(' active', '');
            }
            
            // 显示当前标签页并添加active类
            document.getElementById(tabName).style.display = 'block';
            evt.currentTarget.className += ' active';
        }
    </script>
</body>
</html>
        ''')
    print(f"已创建HTML模板: {template_path}")
except Exception as e:
    print(f"创建HTML模板时出错: {str(e)}")
    traceback.print_exc()

# 创建批量测试页面
try:
    batch_template_path = os.path.join(templates_dir, 'batch_test.html')
    with open(batch_template_path, 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>ABLH批量预测与可视化</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #2c3e50;
            text-align: center;
        }
        .section {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            display: block;
            margin: 20px auto;
        }
        button:hover {
            background-color: #2980b9;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 10px;
            display: none;
        }
        .result-container {
            margin-top: 20px;
            display: none;
        }
        .metrics {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 4px;
            padding: 15px;
            margin: 10px;
            text-align: center;
            min-width: 150px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }
        .chart-container {
            width: 100%;
            height: 400px;
            margin-bottom: 30px;
        }
        .file-input-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .file-input-label {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            margin-top: 10px;
        }
        .file-input-label:hover {
            background-color: #2980b9;
        }
        #file-input {
            display: none;
        }
        #file-name {
            margin-top: 10px;
            font-style: italic;
        }
        .loading {
            text-align: center;
            display: none;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .download-btn {
            background-color: #27ae60;
        }
        .download-btn:hover {
            background-color: #219653;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .table-container {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 20px;
        }
        .nav-link {
            display: inline-block;
            margin: 10px;
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .nav-link:hover {
            background-color: #2980b9;
        }
    </style>
</head>
<body>
    <h1>ABLH批量预测与可视化</h1>
    
    <div style="text-align: center; margin-bottom: 20px;">
        <a href="/" class="nav-link">单条预测</a>
        <a href="/batch_test" class="nav-link">批量预测</a>
    </div>
    
    <div class="section">
        <h2>上传测试数据</h2>
        <div class="file-input-container">
            <p>请上传包含测试数据的CSV或Excel文件。文件必须包含与训练数据相同的特征列。</p>
            <label for="file-input" class="file-input-label">选择文件</label>
            <input type="file" id="file-input" accept=".csv, .xlsx, .xls">
            <div id="file-name"></div>
        </div>
        <button id="predict-btn" disabled>开始批量预测</button>
    </div>
    
    <div id="loading" class="loading">
        <p>正在处理数据，请稍候...</p>
        <div class="spinner"></div>
    </div>
    
    <div id="error" class="error"></div>
    
    <div id="result-container" class="result-container">
        <h2>预测结果</h2>
        
        <div class="metrics">
            <div class="metric-card">
                <div>均方误差 (MSE)</div>
                <div id="mse" class="metric-value">-</div>
            </div>
            <div class="metric-card">
                <div>均方根误差 (RMSE)</div>
                <div id="rmse" class="metric-value">-</div>
            </div>
            <div class="metric-card">
                <div>平均绝对误差 (MAE)</div>
                <div id="mae" class="metric-value">-</div>
            </div>
            <div class="metric-card">
                <div>决定系数 (R²)</div>
                <div id="r2" class="metric-value">-</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="scatter-chart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="residual-chart"></canvas>
        </div>
        
        <div class="table-container">
            <table id="results-table">
                <thead>
                    <tr>
                        <th>序号</th>
                        <th>实际值</th>
                        <th>预测值</th>
                        <th>误差</th>
                    </tr>
                </thead>
                <tbody>
                    <!-- 结果将通过JavaScript动态添加 -->
                </tbody>
            </table>
        </div>
        
        <button id="download-btn" class="download-btn">下载预测结果</button>
    </div>

    <script>
        // 文件输入处理
        const fileInput = document.getElementById('file-input');
        const fileName = document.getElementById('file-name');
        const predictBtn = document.getElementById('predict-btn');
        let fileData = null;
        
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                fileName.textContent = file.name;
                predictBtn.disabled = false;
                
                // 读取文件
                const reader = new FileReader();
                reader.onload = function(event) {
                    fileData = event.target.result;
                };
                reader.readAsArrayBuffer(file);
            } else {
                fileName.textContent = '';
                predictBtn.disabled = true;
                fileData = null;
            }
        });
        
        // 预测按钮处理
        predictBtn.addEventListener('click', async function() {
            if (!fileData) {
                document.getElementById('error').textContent = '请先选择文件';
                document.getElementById('error').style.display = 'block';
                return;
            }
            
            // 显示加载状态
            document.getElementById('loading').style.display = 'block';
            document.getElementById('error').style.display = 'none';
            document.getElementById('result-container').style.display = 'none';
            
            try {
                // 创建FormData对象
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                // 发送请求
                const response = await fetch('/batch_predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('loading').style.display = 'none';
                    return;
                }
                
                // 显示结果
                displayResults(data);
                
                // 隐藏加载状态，显示结果容器
                document.getElementById('loading').style.display = 'none';
                document.getElementById('result-container').style.display = 'block';
            } catch (error) {
                document.getElementById('error').textContent = '处理过程中发生错误，请稍后再试';
                document.getElementById('error').style.display = 'block';
                document.getElementById('loading').style.display = 'none';
            }
        });
        
        // 显示结果函数
        function displayResults(data) {
            // 显示评估指标
            document.getElementById('mse').textContent = data.metrics.mse.toFixed(2);
            document.getElementById('rmse').textContent = data.metrics.rmse.toFixed(2);
            document.getElementById('mae').textContent = data.metrics.mae.toFixed(2);
            document.getElementById('r2').textContent = data.metrics.r2.toFixed(4);
            
            // 绘制散点图
            const scatterCtx = document.getElementById('scatter-chart').getContext('2d');
            new Chart(scatterCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: '预测值 vs 实际值',
                        data: data.predictions.map((p, i) => ({
                            x: data.actual[i],
                            y: p
                        })),
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    }, {
                        label: '理想线',
                        data: (() => {
                            const min = Math.min(...data.actual);
                            const max = Math.max(...data.actual);
                            return [
                                { x: min, y: min },
                                { x: max, y: max }
                            ];
                        })(),
                        type: 'line',
                        borderColor: 'rgba(231, 76, 60, 0.8)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    title: {
                        display: true,
                        text: '预测值 vs 实际值'
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: '实际值'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: '预测值'
                            }
                        }
                    }
                }
            });
            
            // 绘制残差图
            const residuals = data.predictions.map((p, i) => data.actual[i] - p);
            const residualCtx = document.getElementById('residual-chart').getContext('2d');
            new Chart(residualCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: '残差',
                        data: data.predictions.map((p, i) => ({
                            x: p,
                            y: residuals[i]
                        })),
                        backgroundColor: 'rgba(46, 204, 113, 0.7)',
                        borderColor: 'rgba(46, 204, 113, 1)',
                        borderWidth: 1
                    }, {
                        label: '零线',
                        data: (() => {
                            const min = Math.min(...data.predictions);
                            const max = Math.max(...data.predictions);
                            return [
                                { x: min, y: 0 },
                                { x: max, y: 0 }
                            ];
                        })(),
                        type: 'line',
                        borderColor: 'rgba(231, 76, 60, 0.8)',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        pointRadius: 0,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    title: {
                        display: true,
                        text: '残差图'
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: '预测值'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: '残差'
                            }
                        }
                    }
                }
            });
            
            // 填充结果表格
            const tableBody = document.getElementById('results-table').getElementsByTagName('tbody')[0];
            tableBody.innerHTML = '';
            
            data.predictions.forEach((pred, i) => {
                const row = tableBody.insertRow();
                
                const cellIndex = row.insertCell(0);
                cellIndex.textContent = i + 1;
                
                const cellActual = row.insertCell(1);
                cellActual.textContent = data.actual[i].toFixed(2);
                
                const cellPredicted = row.insertCell(2);
                cellPredicted.textContent = pred.toFixed(2);
                
                const cellError = row.insertCell(3);
                const error = data.actual[i] - pred;
                cellError.textContent = error.toFixed(2);
                if (Math.abs(error) > data.metrics.rmse) {
                    cellError.style.color = 'red';
                }
            });
            
            // 下载按钮处理
            document.getElementById('download-btn').addEventListener('click', function() {
                // 创建CSV内容
                let csvContent = "序号,实际值,预测值,误差\n";
                
                data.predictions.forEach((pred, i) => {
                    const error = data.actual[i] - pred;
                    csvContent += `${i+1},${data.actual[i].toFixed(2)},${pred.toFixed(2)},${error.toFixed(2)}\n`;
                });
                
                // 创建下载链接
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'prediction_results.csv';
                link.click();
                URL.revokeObjectURL(url);
            });
        });
    </script>
</body>
</html>
''');
    print("已创建HTML模板:", template_path);
    
    # 创建批量测试HTML模板
    batch_test_path = os.path.join(templates_dir, 'batch_test.html');
    with open(batch_test_path, 'w', encoding='utf-8') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>ABLH批量预测</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* 使用与index.html相同的样式 */
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .file-upload {
            margin-bottom: 20px;
            text-align: center;
        }
        .result {
            margin-top: 20px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            text-align: center;
            font-size: 18px;
            display: none;
        }
        .error {
            color: #e74c3c;
            text-align: center;
            margin-top: 10px;
            display: none;
        }
        .nav-link {
            display: inline-block;
            margin: 10px;
            padding: 8px 15px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
        }
        .nav-link:hover {
            background-color: #2980b9;
        }
        .chart-container {
            margin-top: 20px;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            margin-bottom: 20px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metrics {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-bottom: 20px;
        }
        .metric {
            width: 48%;
            background-color: #f8f9fa;
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }
        .metric-name {
            font-size: 14px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>大气边界层高度(ABLH)批量预测</h1>
    
    <div style="text-align: center; margin-bottom: 20px;">
        <a href="/" class="nav-link">单条预测</a>
        <a href="/batch_test" class="nav-link">批量预测</a>
    </div>
    
    <div class="file-upload">
        <input type="file" id="file-input" accept=".xlsx,.xls,.csv">
        <p>支持的文件格式: Excel (.xlsx, .xls) 或 CSV (.csv)</p>
    </div>
    
    <div id="error" class="error"></div>
    <div id="result" class="result"></div>
    
    <div class="metrics" id="metrics" style="display: none;">
        <div class="metric">
            <div class="metric-value" id="mae">-</div>
            <div class="metric-name">平均绝对误差 (MAE)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="rmse">-</div>
            <div class="metric-name">均方根误差 (RMSE)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="r2">-</div>
            <div class="metric-name">决定系数 (R²)</div>
        </div>
        <div class="metric">
            <div class="metric-value" id="mape">-</div>
            <div class="metric-name">平均绝对百分比误差 (MAPE)</div>
        </div>
    </div>
    
    <div class="chart-container" id="scatter-plot" style="display: none;">
        <h3>预测值与实际值对比图</h3>
        <img id="scatter-plot-image" src="" alt="预测值与实际值对比散点图">
    </div>
    
    <div class="chart-container" id="residual-plot" style="display: none;">
        <h3>残差分布图</h3>
        <img id="residual-plot-image" src="" alt="残差分布图">
    </div>

    <script>
        document.getElementById('file-input').addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            // 显示加载提示
            const resultDiv = document.getElementById('result');
            resultDiv.textContent = '正在处理数据，请稍候...';
            resultDiv.style.display = 'block';
            
            // 隐藏之前的结果
            document.getElementById('metrics').style.display = 'none';
            document.getElementById('scatter-plot').style.display = 'none';
            document.getElementById('residual-plot').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            
            fetch('/batch_predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('error').textContent = data.error;
                    document.getElementById('error').style.display = 'block';
                    resultDiv.style.display = 'none';
                } else {
                    // 显示评估指标
                    document.getElementById('mae').textContent = data.metrics.mae.toFixed(2);
                    document.getElementById('rmse').textContent = data.metrics.rmse.toFixed(2);
                    document.getElementById('r2').textContent = data.metrics.r2.toFixed(3);
                    document.getElementById('mape').textContent = data.metrics.mape.toFixed(2) + '%';
                    document.getElementById('metrics').style.display = 'flex';
                    
                    // 显示散点图
                    document.getElementById('scatter-plot-image').src = 'data:image/png;base64,' + data.scatter_plot;
                    document.getElementById('scatter-plot').style.display = 'block';
                    
                    // 显示残差图
                    document.getElementById('residual-plot-image').src = 'data:image/png;base64,' + data.residual_plot;
                    document.getElementById('residual-plot').style.display = 'block';
                    
                    resultDiv.textContent = '批量预测完成！';
                }
            })
            .catch(error => {
                console.error('批量预测请求出错:', error);
                document.getElementById('error').textContent = '预测请求失败，请检查网络连接或联系管理员。';
                document.getElementById('error').style.display = 'block';
                resultDiv.style.display = 'none';
            });
        });
    </script>
</body>
</html>
''');
    print("已创建批量测试HTML模板:", batch_test_path);

except Exception as e:
    print(f"创建模板时出错: {str(e)}");
    traceback.print_exc();

@app.route('/')
def index():
    return render_template('index.html', features=feature_columns, raw_features=raw_features);

@app.route('/batch_test')
def batch_test():
    return render_template('batch_test.html');

@app.route('/model_status')
def model_status():
    return jsonify({
        'model_loaded': model is not None,
        'message': '模型未加载，请确保已运行XGBoost.py训练模型' if model is None else '模型已加载'
    });

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json();
        
        if not data:
            return jsonify({'error': '未接收到数据'});
        
        # 创建输入数据的DataFrame
        input_data = pd.DataFrame([data['raw_features']]);
        
        # 添加日期列
        input_data['date'] = pd.to_datetime(data['datetime']);
        
        # 进行特征工程
        X, _, _ = engineer_features(input_data, is_training=False);
        
        # 确保特征列的顺序与训练时一致
        if feature_columns:
            missing_cols = set(feature_columns) - set(X.columns);
            for col in missing_cols:
                X[col] = 0;
            X = X[feature_columns];
        
        # 进行预测
        if model is None:
            return jsonify({'error': '模型未加载，请确保已运行XGBoost.py训练模型'});
        
        prediction = model.predict(X)[0];
        
        return jsonify({'prediction': float(prediction)});
        
    except Exception as e:
        print(f"预测时出错: {str(e)}");
        traceback.print_exc();
        return jsonify({'error': f'预测失败: {str(e)}'});

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '未上传文件'});
        
        file = request.files['file'];
        if file.filename == '':
            return jsonify({'error': '未选择文件'});
        
        # 读取文件
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file);
        else:
            df = pd.read_excel(file);
        
        # 进行特征工程
        X, y, _ = engineer_features(df, is_training=True);
        
        # 确保特征列的顺序与训练时一致
        if feature_columns:
            missing_cols = set(feature_columns) - set(X.columns);
            for col in missing_cols:
                X[col] = 0;
            X = X[feature_columns];
        
        # 进行预测
        if model is None:
            return jsonify({'error': '模型未加载，请确保已运行XGBoost.py训练模型'});
        
        predictions = model.predict(X);
        
        # 计算评估指标
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score;
        mae = mean_absolute_error(y, predictions);
        rmse = np.sqrt(mean_squared_error(y, predictions));
        r2 = r2_score(y, predictions);
        mape = np.mean(np.abs((y - predictions) / y)) * 100;
        
        # 创建散点图（更新标题和标签的字体设置）
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.title(f'预测值与实际值对比\nR$^{2}$ = {r2:.2f}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 将图像转换为base64字符串
        scatter_buffer = io.BytesIO()
        plt.savefig(scatter_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        scatter_buffer.seek(0)
        scatter_plot = base64.b64encode(scatter_buffer.getvalue()).decode()
        
        # 创建残差图（更新标题和标签的字体设置）
        residuals = predictions - y
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('预测值')
        plt.ylabel('残差')
        plt.title('残差分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 将残差图转换为base64字符串
        residual_buffer = io.BytesIO()
        plt.savefig(residual_buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        residual_buffer.seek(0)
        residual_plot = base64.b64encode(residual_buffer.getvalue()).decode()
        
        return jsonify({
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'mape': mape
            },
            'scatter_plot': scatter_plot,
            'residual_plot': residual_plot
        });
        
    except Exception as e:
        print(f"批量预测时出错: {str(e)}");
        traceback.print_exc();
        return jsonify({'error': f'批量预测失败: {str(e)}'});

if __name__ == '__main__':
    app.run(debug=True);
                
                