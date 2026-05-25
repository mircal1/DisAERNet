import json
import os
import random

import pandas as pd
import torch
import torchmetrics
import numpy as np

from torch import nn
from torch.optim import RMSprop, Adam
from scipy.stats import pearsonr
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from dataset.dsnet import DSNETDataset

from preprocessing import preprocess_cont_data, preprocess_cate_data, preprocess_cate_backward_data, \
    preprocess_minmax_data


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


# Projector network
# class Projector(nn.Module):
#     def __init__(self, feature_size, projector_size):
#         super(Projector, self).__init__()
#         self.linear = nn.Linear(feature_size, projector_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         return self.sigmoid(self.linear(x))


def load_model(model, file_path, device):
    # 加载模型的状态字典
    model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
    print(f"Model loaded from {file_path}")
    return model

class Projector(nn.Module):
    def __init__(self, feature_size, projector_size, hidden_size, dropout_prob=0.2):
        """
        :param feature_size: 输入特征维度
        :param projector_size: 输出特征维度
        :param hidden_size: 隐藏层维度（可根据需要调整）
        :param dropout_prob: dropout 概率
        """
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_size, projector_size),
            nn.Sigmoid()  
        )
        
    def forward(self, x):
        return self.net(x)


# You would call this function with your dataset, batch size, feature size, and the number of classes
# self_supervised_learning_with_switchtab(data, batch_size, feature_size, num_classes)
def _preprocess_input_data(input_data, train_json):
    cont_data = preprocess_cont_data(
        input_data=input_data[train_json.get("data")["cont_cols"]].copy(),
        file_path=os.path.join("models/standard_scaler.pkl"),
        is_train=True
    )
    cate_data = preprocess_cate_backward_data(
        input_data=input_data[train_json.get("data")["cate_cols"]].copy(),
        file_path=os.path.join("models/backward_encoders.pkl"),
        is_train=True
    )
    return cont_data, cate_data


def _gen_train_dataset(train_data, common_values, train_json):
    x_cont_train, x_cate_train = _preprocess_input_data(train_data, train_json)
    x_cate_common_train = x_cate_train[common_values]

    if (x_cont_train.shape[1] + x_cate_common_train.shape[1]) % 2:
        x_cate_common_train = np.delete(x_cate_common_train, 0, axis=1)
    cont_data = preprocess_minmax_data(x_cont_train)
    cate_data = preprocess_minmax_data(x_cate_common_train)
    train_set = DSNETDataset(cont_data, cate_data,
                             train_data[train_json.get("data")["target_col"]])

    feature_size = train_set.x_cont.shape[1] + train_set.x_cate.shape[1]
    return train_set, feature_size


def train_xgboost_model(test_set, batch_size, feature_size, projector_size):
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    # Initialize the components with the feature size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed(42)
    hidden_size=config['model']['hidden_size']
    pm_mutual = Projector(feature_size, projector_size,hidden_size).to(device)
    ps_salient = Projector(feature_size, projector_size,hidden_size).to(device)
    loaded_pre_encoder = Encoder(feature_size=feature_size).to(device)

    f_encoder = load_model(loaded_pre_encoder, "models/pre_encoder.pth", device)
    s_projector = load_model(ps_salient, "models/s_projector.pth", device)
    m_projector = load_model(pm_mutual, "models/m_projector.pth", device)

    feature_batchs = []
    for x_batch, labels in test_dataloader:
        x_batch = x_batch.to(device)
        labels = labels.to(device)
        # Assume that now we have labels
        z_encoded = f_encoder(x_batch)
        s_salient = s_projector(z_encoded)
        m_mutual = m_projector(z_encoded)
        s_m_batch = torch.concat((s_salient, m_mutual), dim=1)
        feature_batchs.append(s_m_batch)
    return torch.cat(feature_batchs, dim=0).cpu().detach().numpy()


if __name__ == "__main__":
    with open('train.json', 'r') as f:
        config = json.load(f)
    with open("common_columns.json", 'r') as f:
        common_columns = json.load(f)

    train_data = pd.read_csv(config.get("data")["train_data"])
    valid_data= pd.read_csv(config.get("data")["valid_data"])
    test_data= pd.read_csv(config.get("data")["test_data"])
    common_columns = sorted(list(common_columns))

    batch_size = config.get("dataloader")["batch_size"]
    projector_size = config.get("model")["projector_size"]
    train_set, feature_size = _gen_train_dataset(train_data, common_columns, config)

    feature_data = train_xgboost_model(train_set, batch_size, feature_size, projector_size)
    # 转为csv
    feature_columns = ["PROJECTOR_" + str(i) for i in range(projector_size * 2)]
    train_data[feature_columns] = feature_data
    train_name = os.path.basename(config.get("data")["train_data"]).split(".")[0]
    train_folder = config.get("global")["output_folder"]
    # if not os.path.exists(train_folder):
    #     os.makedirs(train_folder)
    train_data.to_csv(os.path.join(train_folder, f"{train_name}_projector.csv"), index=False)
   
    valid_set, feature_size = _gen_train_dataset(valid_data, common_columns, config)
    feature_data = train_xgboost_model(valid_set, batch_size, feature_size, projector_size)
    valid_data[feature_columns] = feature_data
    valid_name = os.path.basename(config.get("data")["valid_data"]).split(".")[0]
    valid_folder = config.get("global")["output_folder"]
    valid_data.to_csv(os.path.join(valid_folder, f"{valid_name}_projector.csv"), index=False)

    test_set, feature_size = _gen_train_dataset(test_data, common_columns, config)
    feature_data = train_xgboost_model(test_set, batch_size, feature_size, projector_size)
    test_data[feature_columns] = feature_data
    test_name = os.path.basename(config.get("data")["test_data"]).split(".")[0]
    test_folder = config.get("global")["output_folder"]
    test_data.to_csv(os.path.join(test_folder, f"{test_name}_projector.csv"), index=False)

    print("success save new feature data to", train_folder, f"{train_name}_projector.csv",f"{valid_name}_projector.csv",f"{test_name}_projector.csv")
