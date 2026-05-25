import json
import os

import pandas as pd
import numpy as np
import torch

from torch import nn
from scipy.stats import pearsonr
from scipy.stats import linregress
from dataset.dsnet import DSNETDataset
import torchmetrics

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import preprocess_cont_data, preprocess_cate_data, preprocess_minmax_data, \
    preprocess_cate_backward_data


# Encoder network with a three-layer transformer
class Encoder(nn.Module):
    def __init__(self, feature_size, num_heads=2):
        super(Encoder, self).__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads)
        )

    def forward(self, x):
        # Since Transformer expects seq_length x batch x features, we assume x is already shaped correctly
        return self.transformer_layers(x)


class MLPBlock(nn.Module):

    def __init__(self, d_inp: int, d_out: int, p_drop=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_inp, d_out),
            nn.Sigmoid(),
            nn.BatchNorm1d(d_out),
            nn.Dropout(p_drop)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

class Predictor(nn.Module):
    def __init__(self, feature_size, hidden_dim, num_classes, model_config):
        super(Predictor, self).__init__()
        n_layers = model_config['n_layers']
        p_drop = model_config["p_drop"]
        self.input_layer = MLPBlock(feature_size, hidden_dim, p_drop)
        self.hidden_layers = nn.Sequential(
            *[MLPBlock(hidden_dim, hidden_dim, p_drop) for _ in range(n_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

# 推理函数
def inference(data, batch_size, feature_size, num_classes,model_config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")

    loaded_f_encoder = Encoder(feature_size=feature_size).to(device)
    f_encoder = load_model(loaded_f_encoder, "models/f_encoder.pth", device)

    loaded_pred_predictor = Predictor(feature_size, feature_size//4, num_classes,model_config).to(device)
    pred_predictor = load_model(loaded_pred_predictor, "models/pred_predictor.pth", device)

    f_encoder.eval()  # 设置为评估模式
    pred_predictor.eval()  # 设置为评估模式
    fine_tuning_loss_function = nn.MSELoss()
    total_loss = 0
    all_predictions = []
    all_labels = []
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    with torch.no_grad():  # 关闭梯度计算
        for x_batch, labels in dataloader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            z_encoded = f_encoder(x_batch)
            predictions = pred_predictor(z_encoded)

            labels = labels.unsqueeze(1)

            loss = fine_tuning_loss_function(predictions, labels)
            total_loss += loss.item()

            all_predictions.append(predictions)
            all_labels.append(labels)

    # 合并所有 batch 的预测值和标签
    all_predictions = torch.cat(all_predictions, dim=0).flatten().cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).flatten().cpu().numpy()
    # 统计
    print(f'test data:',calc_regression_metric(all_labels, all_predictions))
    # 计算平均损失
    average_loss = total_loss / len(dataloader)
    #print(f'Inference Loss: {average_loss:.4f}')

    return all_predictions, all_labels, average_loss


def calc_regression_metric(y_true, y_pred):
    return {
        #"COUNT": len(y_true),
        #"RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        #"MAE": mean_absolute_error(y_true, y_pred),
        "R2": linregress(y_true, y_pred).rvalue**2,
        #"PEARSON": pearsonr(y_true, y_pred)[0]
    }


# self_supervised_learning_with_switchtab(data, batch_size, feature_size, num_classes)
def _preprocess_input_data(input_data, config):
    cont_data = preprocess_cont_data(
        input_data=input_data[config.get("data")["cont_cols"]].copy(),
        file_path=os.path.join("models/standard_scaler.pkl"),
        is_train=True
    )
    cate_data = preprocess_cate_backward_data(
        input_data=input_data[config.get("data")["cate_cols"]].copy(),
        file_path=os.path.join("models/backward_encoders.pkl"),
        is_train=True
    )
    return cont_data, cate_data


def _gen_test_dataset(test_data, common_values, inference_json):
    x_cont_test, x_cate_test = _preprocess_input_data(test_data, inference_json)
    x_cate_common_test = x_cate_test[common_values]

    if (x_cont_test.shape[1] + x_cate_common_test.shape[1]) % 2:
        x_cate_common_test = np.delete(x_cate_common_test, 0, axis=1)
    cont_data = preprocess_minmax_data(x_cont_test)
    cate_data = preprocess_minmax_data(x_cate_common_test)
    test_set = DSNETDataset(cont_data, cate_data,
                            test_data[inference_json.get("data")["target_col"]])

    feature_size = test_set.x_cont.shape[1] + test_set.x_cate.shape[1]
    return test_set, feature_size


def load_model(model, file_path, device):
    # 加载模型的状态字典
    model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
    model.eval()  # 切换模型为评估模式
    #print(f"Model loaded from {file_path}")
    return model


if __name__ == "__main__":

    with open("inference.json", "r") as f:
        inference_json = json.load(f)
    with open("common_columns.json", 'r') as f:
        common_columns = json.load(f)
    test_data = pd.read_csv(inference_json.get("data")["inference"])
    common_columns = sorted(list(common_columns))
    test_set, feature_size = _gen_test_dataset(test_data, common_columns, inference_json)
    batch_size = inference_json.get("dataloader")["batch_size"]
    model_config = inference_json.get("model")["net"]
    predictions, true_labels, inference_loss = inference(test_set, batch_size, feature_size, 1,model_config)
    # 输出预测结果和真实标签（可视化或评估）
    print("Predictions:\n", predictions)
    print("True Labels:\n", true_labels)
    # 转为csv
    test_data[inference_json.get("data")["target_col"] + "_PRED"] = predictions
    test_name = os.path.basename(inference_json.get("data")["inference"]).split(".")[0]
    test_folder = inference_json.get("global")["output_folder"]
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    test_data.to_csv(os.path.join(test_folder, f"{test_name}_pred.csv"), index=False)
