import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# train_data = pd.read_csv("./data/train_land.csv")
# valid_data= pd.read_csv("./data/valid_land.csv")
# test_data=pd.read_csv("./data/test_land.csv")

train_data = pd.read_csv("./densenet/train/train_land_projector.csv")
valid_data= pd.read_csv("./densenet/train/valid_land_projector.csv")
test_data=pd.read_csv("./densenet/train/test_land_projector.csv")

# X_columns = [
#     "Lt_VN01", "Lt_VN02", "Lt_VN03", "Lt_VN04", "Lt_VN05", "Lt_VN06", "Lt_VN07", "Lt_VN08", "Lt_VN08P", "Lt_VN09", 
#     "Lt_VN10", "Lt_VN11", "Lt_VN11P", "Lt_SW01", "Lt_SW02", "Lt_SW03", "Lt_SW04", "Lt_TI01", "Lt_TI02", "scatter_ag", 
#     "Sensor_zenith", "Solar_zenith", "scatter_ag_PL", "Sensor_zenith_PL", "Solar_zenith_PL", "Lt_PI01", "Lt_PI02", 
#     "Lt_PQ01", "Lt_PQ02", "Lt_PU01", "Lt_PU02", "Lt_P1_0", "Lt_P2_0", "Lt_P1_m60", "Lt_P2_m60", "Lt_P1_p60", 
#     "Lt_P2_p60", "scatter_ag_IR", "Sensor_zenith_IR", "MNDWI", "NDVI", "BSI", "NBDI", "NDSI", "NDISI","month","season"
# ]

X_columns = [
    "Lt_VN01", "Lt_VN02", "Lt_VN03", "Lt_VN04", "Lt_VN05", "Lt_VN06", "Lt_VN07", "Lt_VN08", "Lt_VN08P", "Lt_VN09", 
    "Lt_VN10", "Lt_VN11", "Lt_VN11P", "Lt_SW01", "Lt_SW02", "Lt_SW03", "Lt_SW04", "Lt_TI01", "Lt_TI02", "scatter_ag", 
    "Sensor_zenith", "Solar_zenith", "scatter_ag_PL", "Sensor_zenith_PL", "Solar_zenith_PL", "Lt_PI01", "Lt_PI02", 
    "Lt_PQ01", "Lt_PQ02", "Lt_PU01", "Lt_PU02", "Lt_P1_0", "Lt_P2_0", "Lt_P1_m60", "Lt_P2_m60", "Lt_P1_p60", 
    "Lt_P2_p60", "scatter_ag_IR", "Sensor_zenith_IR", "MNDWI", "NDVI", "BSI", "NBDI", "NDSI", "NDISI","month","season",
    "PROJECTOR_0","PROJECTOR_1"
    #,"PROJECTOR_2"
    #,"PROJECTOR_3","PROJECTOR_4","PROJECTOR_5","PROJECTOR_6","PROJECTOR_7"
    #,"PROJECTOR_8","PROJECTOR_9","PROJECTOR_10","PROJECTOR_11"
    #,"PROJECTOR_12","PROJECTOR_13","PROJECTOR_14","PROJECTOR_15"
    #,"PROJECTOR_16","PROJECTOR_17","PROJECTOR_18","PROJECTOR_19"
]

# 从 train_data 中选择 X 和 Y
X_train = train_data[X_columns]  # 特征列
y_train = train_data['fAOD']  # 目标变量

# 从 valid_data 和 test_data 中选择 X 和 Y
X_valid = valid_data[X_columns]
y_valid = valid_data['fAOD']

X_test = test_data[X_columns]
y_test = test_data['fAOD']


# 转换为 DMatrix 格式，这是 XGBoost 推荐的格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置 XGBoost 超参数
params = {
    'objective': 'reg:squarederror',  # 回归问题，若是分类问题可改为 'binary:logistic'
    'tree_method': 'gpu_hist',  # 使用 GPU 加速
    'gpu_id': 0,  # 选择 GPU ID
    'eval_metric': 'rmse',  # 用于回归问题的评估指标
    'max_depth': 6,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1
}

# 用于调整超参数的列表
param_grid = {
    'max_depth': [4, 6, 8],
    'eta': [0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8],
    'lambda': [1, 1.5,2]
}

# 调整超参数的逻辑
best_rmse = float('inf')
best_params = None

for depth in param_grid['max_depth']:
    for eta in param_grid['eta']:
        for subsample in param_grid['subsample']:
            for colsample in param_grid['colsample_bytree']:
                for lmbda in param_grid['lambda']:
                    params.update({
                        'max_depth': depth,
                        'eta': eta,
                        'subsample': subsample,
                        'colsample_bytree': colsample,
                        'lambda': lmbda
                    })

                    # 设置训练过程中的验证集
                    evals = [(dtrain, 'train'), (dvalid, 'valid')]

                    # 训练模型
                    model = xgb.train(params, dtrain, num_boost_round=1000, evals=evals, early_stopping_rounds=50)

                    # 在验证集上计算 RMSE
                    y_pred_valid = model.predict(dvalid)
                    rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))  # 修正为 squared=False

                    # 如果 RMSE 提升，更新最佳参数
                    if rmse_valid < best_rmse:
                        best_rmse = rmse_valid
                        best_params = {
                            'max_depth': depth,
                            'eta': eta,
                            'subsample': subsample,
                            'colsample_bytree': colsample,
                            'lambda': lmbda
                        }

# 使用最佳参数进行最终模型训练
params.update(best_params)
model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dtrain, 'train'), (dvalid, 'valid')], early_stopping_rounds=50)

# 在测试集上进行预测
y_pred_test = model.predict(dtest)

# 计算 RMSE 和 R²
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))  # 修正为 squared=False
r2_test = r2_score(y_test, y_pred_test)

print(f'RMSE on Test Data: {rmse_test}')
print(f'R² on Test Data: {r2_test}')

# 可视化重要特征
xgb.plot_importance(model)
plt.show()