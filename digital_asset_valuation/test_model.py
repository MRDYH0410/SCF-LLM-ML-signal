import joblib

# 加载模型
model = joblib.load("valuation_model_lgbm.pkl")

# 使用模型进行预测示例
X_sample = [[0.6, 0.7, 0.5, 0.65, 0.55, 0.12, 0.4, 0.03, 130]]  # 替换为你的特征输入
y_pred = model.predict(X_sample)

print(f"预测结果: {y_pred}")
print(model)  # 打印模型参数
print(model.feature_importances_)  # 显示特征重要性