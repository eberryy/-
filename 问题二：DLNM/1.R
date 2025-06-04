# 加载必要的包
library(dlnm)      # 分布式滞后非线性模型
library(splines)   # 样条函数
library(mgcv)      # 广义加性模型
library(ggplot2)   # 可视化
library(gridExtra) # 多图布局

# 假设数据已经导入为"第二问数据"
# 确保日期列为日期格式
第二问数据$date <- as.numeric(第二问数据$date)  # 先转为数值
第二问数据$date <- as.Date(第二问数据$date, origin="2023-01-01")  # 假设起始日期，请根据实际情况调整

# 检查数据结构
str(第二问数据)
summary(第二问数据)

# 定义滞后天数范围
lag_max <- 7  # 最大滞后天数，可根据实际情况调整

# 为ABLH创建交叉基矩阵
# 使用自然样条函数建模ABLH的非线性效应
# 使用自然样条函数建模滞后效应
cb_ablh <- crossbasis(
  第二问数据$ABLH, 
  lag = lag_max,
  argvar = list(fun = "ns", df = 3),  # ABLH的非线性效应，使用3个自由度
  arglag = list(fun = "ns", df = 3)   # 滞后效应，使用3个自由度
)

# 分析ABLH与PM2.5的关系
model_pm25 <- glm(PM2.5 ~ cb_ablh, family = gaussian(), data = 第二问数据)

# 分析ABLH与PM10的关系
model_pm10 <- glm(PM10 ~ cb_ablh, family = gaussian(), data = 第二问数据)

# 分析ABLH与NO2的关系
model_no2 <- glm(NO2 ~ cb_ablh, family = gaussian(), data = 第二问数据)

# 分析ABLH与AQI的关系
model_aqi <- glm(AQI ~ cb_ablh, family = gaussian(), data = 第二问数据)

# 预测和可视化结果
# 定义ABLH的预测值范围
pred_ablh <- seq(min(第二问数据$ABLH), max(第二问数据$ABLH), length.out = 50)

# 为PM2.5创建预测
pred_pm25 <- crosspred(cb_ablh, model_pm25, at = pred_ablh, bylag = 0.2, cumul = TRUE)

# 为PM10创建预测
pred_pm10 <- crosspred(cb_ablh, model_pm10, at = pred_ablh, bylag = 0.2, cumul = TRUE)

# 为NO2创建预测
pred_no2 <- crosspred(cb_ablh, model_no2, at = pred_ablh, bylag = 0.2, cumul = TRUE)

# 为AQI创建预测
pred_aqi <- crosspred(cb_ablh, model_aqi, at = pred_ablh, bylag = 0.2, cumul = TRUE)









# 可视化结果
# 3D图：ABLH、滞后天数和PM2.5的关系
plot_pm25_3d <- plot(pred_pm25, xlab = "ABLH", ylab = "Lag days", zlab = "PM2.5 effect",
                     main = "Nonlinear lag effect of ABLH on PM2.5", theta = 30, phi = 30)

plot_pm10_3d <- plot(pred_pm10, xlab = "ABLH", ylab = "Lag days", zlab = "PM10 effect",
                     main = "Nonlinear lag effect of ABLH on PM10", theta = 30, phi = 30)

plot_no2_3d <- plot(pred_no2, xlab = "ABLH", ylab = "Lag days", zlab = "NO2 effect",
                    main = "Nonlinear lag effect of ABLH on NO2", theta = 30, phi = 30)

plot_aqi_3d <- plot(pred_aqi, xlab = "ABLH", ylab = "Lag days", zlab = "AQI effect",
                    main = "Nonlinear lag effect of ABLH on AQI", theta = 30, phi = 30)










# 更改为使用固定的ABLH值进行切片图分析

# 首先分析各指标与ABLH的关系，找出关键ABLH值
# 查看所有指标的累积效应曲线
par(mfrow = c(2, 2))
plot(pred_pm25, "overall", xlab = "ABLH", ylab = "Cumulative Effect", main = "Cumulative Effects of ABLH on PM2.5")
plot(pred_pm10, "overall", xlab = "ABLH", ylab = "Cumulative Effect", main = "Cumulative Effects of ABLH on PM10")
plot(pred_no2, "overall", xlab = "ABLH", ylab = "Cumulative Effect", main = "Cumulative Effects of ABLH on NO2")
plot(pred_aqi, "overall", xlab = "ABLH", ylab = "Cumulative Effect", main = "Cumulative Effects of ABLH on AQI")
par(mfrow = c(1, 1))

# 使用固定的ABLH值：200米、400米、600米
# 确保这些值在pred_ablh中存在，找到最接近的实际值
find_closest <- function(value, vector) {
  vector[which.min(abs(vector - value))]
}

# 设置固定的ABLH值
fixed_ablh_values <- c(200, 400, 600)

# 为每个指标找到最接近的实际ABLH值
ablh_values_pm25 <- sapply(fixed_ablh_values, find_closest, pred_ablh)
ablh_values_pm10 <- sapply(fixed_ablh_values, find_closest, pred_ablh)
ablh_values_no2 <- sapply(fixed_ablh_values, find_closest, pred_ablh)
ablh_values_aqi <- sapply(fixed_ablh_values, find_closest, pred_ablh)

# 打印实际使用的ABLH值
print(paste("PM2.5的ABLH值:", paste(round(ablh_values_pm25), collapse=", ")))
print(paste("PM10的ABLH值:", paste(round(ablh_values_pm10), collapse=", ")))
print(paste("NO2的ABLH值:", paste(round(ablh_values_no2), collapse=", ")))
print(paste("AQI的ABLH值:", paste(round(ablh_values_aqi), collapse=", ")))

# PM2.5的切片图 - 使用固定的ABLH值
par(mfrow = c(1, 1))
plot_pm25_lag <- plot(pred_pm25, "slices", var = ablh_values_pm25, col = c("black", "red", "green"),
                      xlab = "Lag (days)", ylab = "PM2.5 effect",
                      main = "Lag effects of different ABLH values on PM2.5",
                      ci.arg = list(col = "grey90", lwd = 2))



# PM10的切片图 - 使用固定的ABLH值
par(mfrow = c(1, 1))
plot_pm10_lag <- plot(pred_pm10, "slices", var = ablh_values_pm10, col = c("black", "red", "green"),
                      xlab = "Lag (days)", ylab = "PM10 effect",
                      main = "Lag effects of different ABLH values on PM10",
                      ci.arg = list(col = "grey90", lwd = 2))


# NO2的切片图 - 使用固定的ABLH值
par(mfrow = c(1, 1))
plot_no2_lag <- plot(pred_no2, "slices", var = ablh_values_no2, col = c("black", "red", "green"),
                     xlab = "Lag (days)", ylab = "NO2 effect",
                     main = "Lag effects of different ABLH values on NO2",
                     ci.arg = list(col = "grey90", lwd = 2))


# AQI的切片图 - 使用固定的ABLH值
par(mfrow = c(1, 1))
plot_aqi_lag <- plot(pred_aqi, "slices", var = ablh_values_aqi, col = c("black", "red", "green"),
                     xlab = "Lag (days)", ylab = "AQI effect",
                     main = "Lag effects of different ABLH values on AQI",
                     ci.arg = list(col = "grey90", lwd = 2))























#为四个因素分别绘制滞后效应图

# 1. PM2.5的滞后效应图
par(mfrow = c(1, 1))  # 重置为单图布局

# 选择代表性的滞后天数
lag_values <- c(0, 1, 3, 5)

# 首先，检查矩阵中实际可用的列名
colnames_available_pm25 <- colnames(pred_pm25$matfit)

# 从列名中提取数值，去除"lag"前缀
lag_numeric_pm25 <- as.numeric(gsub("lag", "", colnames_available_pm25))

# 创建空白图框，使用第一个滞后值
plot(pred_ablh, pred_pm25$matfit[, 1], type = "n",
     xlab = "ABLH", ylab = "PM2.5 effect",
     main = "Effects of ABLH on PM2.5 at different lag days",
     ylim = range(pred_pm25$matfit, na.rm = TRUE))

# 为不同滞后天数添加曲线
colors <- c("red", "blue", "green", "purple")  # 为不同滞后天数定义颜色
legend_text_pm25 <- character(length(lag_values))

for (i in 1:length(lag_values)) {
  # 找到可用列中最接近的滞后值
  lag_diff <- abs(lag_numeric_pm25 - lag_values[i])
  closest_idx <- which.min(lag_diff)
  
  actual_lag <- colnames_available_pm25[closest_idx]
  lines(pred_ablh, pred_pm25$matfit[, actual_lag], col = colors[i], lwd = 2)
  legend_text_pm25[i] <- paste("Lag =", gsub("lag", "", actual_lag), "days")
}

# 添加参考线
abline(h = 0, lty = 2)

# 添加图例
legend("topright", legend = legend_text_pm25,
       col = colors, lty = 1, lwd = 2)

# 2. PM10的滞后效应图
par(mfrow = c(1, 1))  # 重置为单图布局

# 检查PM10矩阵中实际可用的列名
colnames_available_pm10 <- colnames(pred_pm10$matfit)

# 从列名中提取数值，去除"lag"前缀
lag_numeric_pm10 <- as.numeric(gsub("lag", "", colnames_available_pm10))

# 创建空白图框，使用第一个滞后值
plot(pred_ablh, pred_pm10$matfit[, 1], type = "n",
     xlab = "ABLH", ylab = "PM10 effect",
     main = "Effects of ABLH on PM10 at different lag days",
     ylim = range(pred_pm10$matfit, na.rm = TRUE))

# 为不同滞后天数添加曲线
legend_text_pm10 <- character(length(lag_values))

for (i in 1:length(lag_values)) {
  # 找到可用列中最接近的滞后值
  lag_diff <- abs(lag_numeric_pm10 - lag_values[i])
  closest_idx <- which.min(lag_diff)
  
  actual_lag <- colnames_available_pm10[closest_idx]
  lines(pred_ablh, pred_pm10$matfit[, actual_lag], col = colors[i], lwd = 2)
  legend_text_pm10[i] <- paste("Lag =", gsub("lag", "", actual_lag), "days")
}

# 添加参考线
abline(h = 0, lty = 2)

# 添加图例
legend("topright", legend = legend_text_pm10,
       col = colors, lty = 1, lwd = 2)

# 3. NO2的滞后效应图
par(mfrow = c(1, 1))  # 重置为单图布局

# 检查NO2矩阵中实际可用的列名
colnames_available_no2 <- colnames(pred_no2$matfit)

# 从列名中提取数值，去除"lag"前缀
lag_numeric_no2 <- as.numeric(gsub("lag", "", colnames_available_no2))

# 创建空白图框，使用第一个滞后值
plot(pred_ablh, pred_no2$matfit[, 1], type = "n",
     xlab = "ABLH", ylab = "NO2 effect",
     main = "Effects of ABLH on NO2 at different lag days",
     ylim = range(pred_no2$matfit, na.rm = TRUE))

# 为不同滞后天数添加曲线
legend_text_no2 <- character(length(lag_values))

for (i in 1:length(lag_values)) {
  # 找到可用列中最接近的滞后值
  lag_diff <- abs(lag_numeric_no2 - lag_values[i])
  closest_idx <- which.min(lag_diff)
  
  actual_lag <- colnames_available_no2[closest_idx]
  lines(pred_ablh, pred_no2$matfit[, actual_lag], col = colors[i], lwd = 2)
  legend_text_no2[i] <- paste("Lag =", gsub("lag", "", actual_lag), "days")
}

# 添加参考线
abline(h = 0, lty = 2)

# 添加图例
legend("topright", legend = legend_text_no2,
       col = colors, lty = 1, lwd = 2)

# 4. AQI的滞后效应图
par(mfrow = c(1, 1))  # 重置为单图布局

# 检查AQI矩阵中实际可用的列名
colnames_available_aqi <- colnames(pred_aqi$matfit)

# 从列名中提取数值，去除"lag"前缀
lag_numeric_aqi <- as.numeric(gsub("lag", "", colnames_available_aqi))

# 创建空白图框，使用第一个滞后值
plot(pred_ablh, pred_aqi$matfit[, 1], type = "n",
     xlab = "ABLH", ylab = "AQI effect",
     main = "Effects of ABLH on AQI at different lag days",
     ylim = range(pred_aqi$matfit, na.rm = TRUE))

# 为不同滞后天数添加曲线
legend_text_aqi <- character(length(lag_values))

for (i in 1:length(lag_values)) {
  # 找到可用列中最接近的滞后值
  lag_diff <- abs(lag_numeric_aqi - lag_values[i])
  closest_idx <- which.min(lag_diff)
  
  actual_lag <- colnames_available_aqi[closest_idx]
  lines(pred_ablh, pred_aqi$matfit[, actual_lag], col = colors[i], lwd = 2)
  legend_text_aqi[i] <- paste("Lag =", gsub("lag", "", actual_lag), "days")
}

# 添加参考线
abline(h = 0, lty = 2)

# 添加图例
legend("topright", legend = legend_text_aqi,
       col = colors, lty = 1, lwd = 2)


















# 增强模型：考虑温度、湿度等混杂变量
# 假设数据集中有温度(temp)和湿度(humidity)变量
# 如果没有这些变量，可以省略这部分代码
if("temp" %in% names(第二问数据) && "humidity" %in% names(第二问数据)) {
  # 创建温度和湿度的交叉基矩阵
  cb_temp <- crossbasis(
    第二问数据$temp, 
    lag = lag_max,
    argvar = list(fun = "ns", df = 3),
    arglag = list(fun = "ns", df = 2)
  )
  
  cb_humidity <- crossbasis(
    第二问数据$humidity, 
    lag = lag_max,
    argvar = list(fun = "ns", df = 3),
    arglag = list(fun = "ns", df = 2)
  )
  
  # 增强模型，包含温度和湿度
  model_pm25_enhanced <- glm(PM2.5 ~ cb_ablh + cb_temp + cb_humidity, 
                             family = gaussian(), data = 第二问数据)
  
  # 预测和可视化增强模型结果
  pred_pm25_enhanced <- crosspred(cb_ablh, model_pm25_enhanced, at = pred_ablh, bylag = 0.2)
  
  # 比较基础模型和增强模型
  par(mfrow = c(1, 2))
  plot(pred_pm25, "overall", xlab = "ABLH", ylab = "累积效应",
       main = "基础模型：ABLH对PM2.5的累积效应")
  plot(pred_pm25_enhanced, "overall", xlab = "ABLH", ylab = "累积效应",
       main = "增强模型：ABLH对PM2.5的累积效应")
  par(mfrow = c(1, 1))
}






# 模型诊断
par(mfrow = c(2, 2))
plot(model_pm25)
mtext("PM2.5 Model Diagnostic Plot", side = 3, line = -1, outer = TRUE)  # line parameter controls text position, negative value makes it closer to the plot
par(mfrow = c(1, 1))
# Model Diagnostics
par(mfrow = c(2, 2))
plot(model_pm10)
mtext("PM10 Model Diagnostic Plot", side = 3, line = -1, outer = TRUE)
par(mfrow = c(1, 1))
# Model Diagnostics
par(mfrow = c(2, 2))
plot(model_no2)
mtext("NO2 Model Diagnostic Plot", side = 3, line = -1, outer = TRUE)
par(mfrow = c(1, 1))
# Model Diagnostics
par(mfrow = c(2, 2))
plot(model_aqi)
mtext("AQI Model Diagnostic Plot", side = 3, line = -1, outer = TRUE)
par(mfrow = c(1, 1))







# 模型结果汇总
summary(model_pm25)
summary(model_pm10)
summary(model_no2)
summary(model_aqi)

# 计算不同ABLH阈值下的相对风险
# 选择参考值（通常为中位数或特定百分位数）
ablh_ref <- median(第二问数据$ABLH)

# 计算相对风险
rr_pm25 <- crosspred(cb_ablh, model_pm25, at = pred_ablh, cen = ablh_ref)
rr_pm10 <- crosspred(cb_ablh, model_pm10, at = pred_ablh, cen = ablh_ref)
rr_no2 <- crosspred(cb_ablh, model_no2, at = pred_ablh, cen = ablh_ref)
rr_aqi <- crosspred(cb_ablh, model_aqi, at = pred_ablh, cen = ablh_ref)

# 可视化相对风险
# Visualize Relative Risk
plot(rr_pm25, "overall", xlab = "ABLH", ylab = "Relative Risk",
     main = "Relative Risk of ABLH on PM2.5 (Reference: Median)")
abline(h = 1, lty = 2)

plot(rr_pm10, "overall", xlab = "ABLH", ylab = "Relative Risk",
     main = "Relative Risk of ABLH on PM10 (Reference: Median)")
abline(h = 1, lty = 2)

plot(rr_no2, "overall", xlab = "ABLH", ylab = "Relative Risk",
     main = "Relative Risk of ABLH on NO2 (Reference: Median)")
abline(h = 1, lty = 2)

plot(rr_aqi, "overall", xlab = "ABLH", ylab = "Relative Risk",
     main = "Relative Risk of ABLH on AQI (Reference: Median)")
abline(h = 1, lty = 2)

# 保存结果
# 创建结果数据框
results <- data.frame(
  ABLH = pred_ablh,
  PM25_Effect = rr_pm25$allRRfit,
  PM25_Lower = rr_pm25$allRRlow,
  PM25_Upper = rr_pm25$allRRhigh,
  PM10_Effect = rr_pm10$allRRfit,
  PM10_Lower = rr_pm10$allRRlow,
  PM10_Upper = rr_pm10$allRRhigh,
  NO2_Effect = rr_no2$allRRfit,
  NO2_Lower = rr_no2$allRRlow,
  NO2_Upper = rr_no2$allRRhigh,
  AQI_Effect = rr_aqi$allRRfit,
  AQI_Lower = rr_aqi$allRRlow,
  AQI_Upper = rr_aqi$allRRhigh
)

# 保存为CSV文件
write.csv(results, "ABLH_AirQuality_DLNM_Results.csv", row.names = FALSE)

# 使用ggplot2创建更美观的可视化
# 准备数据
pm25_slice_data <- data.frame(
  ABLH = rep(pred_ablh, length(lag_values)),
  Lag = rep(lag_values, each = length(pred_ablh)),
  Effect = unlist(lapply(lag_values, function(l) pred_pm25$matfit[, as.character(l)]))
)

# 创建图形
ggplot_pm25 <- ggplot(pm25_slice_data, aes(x = ABLH, y = Effect, color = factor(Lag))) +
  geom_line() +
  labs(title = "不同滞后天数下ABLH对PM2.5的效应",
       x = "ABLH (m)",
       y = "PM2.5效应",
       color = "滞后天数") +
  theme_minimal() +
  theme(legend.position = "bottom")

# 显示图形
print(ggplot_pm25)

# 创建所有指标的累积效应比较图
all_effects <- data.frame(
  ABLH = rep(pred_ablh, 4),
  Indicator = factor(rep(c("PM2.5", "PM10", "NO2", "AQI"), each = length(pred_ablh))),
  Effect = c(pred_pm25$allfit, pred_pm10$allfit, pred_no2$allfit, pred_aqi$allfit)
)

# Create Comparison Plot
# Create more advanced cumulative effect comparison plot
ggplot_all <- ggplot(all_effects, aes(x = ABLH, y = Effect, color = Indicator)) +
  # Use thicker lines and add points
  geom_line(size = 1.2) +
  # Add points to enhance readability
  geom_point(aes(shape = Indicator), size = 2.5, alpha = 0.7) +
  # Use a more attractive color palette
  scale_color_brewer(palette = "Set1") +
  # Set title and axis labels with larger font
  labs(title = "Comparison of Cumulative Effects of ABLH on Air Quality Indicators",
       x = "Atmospheric Boundary Layer Height (m)",
       y = "Cumulative Effect",
       color = "Air Quality Indicator",
       shape = "Air Quality Indicator") +
  # Use a more attractive theme
  theme_light() +
  # Customize theme elements
  theme(
    # Center title and increase font size
    plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
    # Increase axis label font size
    axis.title = element_text(size = 12, face = "bold"),
    # Increase axis tick font size
    axis.text = element_text(size = 10),
    # Place legend at bottom and adjust style
    legend.position = "bottom",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    # Add grid lines to enhance readability
    panel.grid.major = element_line(color = "gray90", size = 0.3),
    panel.grid.minor = element_line(color = "gray95", size = 0.2),
    # Add border
    panel.border = element_rect(color = "gray80", fill = NA, size = 0.5)
  ) +
  # Add reference line
  geom_hline(yintercept = 0, linetype = "dashed", color = "darkgray", size = 0.5) +
  # Add annotation
  annotate("text", x = max(all_effects$ABLH) * 0.9, y = max(all_effects$Effect) * 0.9,
           label = "Positive values indicate increasing effect\nNegative values indicate decreasing effect",
           size = 3.5, hjust = 1, fontface = "italic")

# Display comparison plot
print(ggplot_all)

# Save as high-quality image
ggsave("ABLH_AirQuality_Cumulative_Effects.png", ggplot_all,
       width = 10, height = 7, dpi = 300)

# Display comparison plot
print(ggplot_all)

