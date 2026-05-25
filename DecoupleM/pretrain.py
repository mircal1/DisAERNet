import json
import os
import random

import pandas as pd
import torch
import numpy as np

from torch import nn
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
from dataset.dsnet import DSNETDataset
from label_pretrainer import LabelPretrainer
from feature_attention import FeatureAttention
from preprocessing import preprocess_cont_data, preprocess_cate_backward_data, \
    preprocess_minmax_data

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(module):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            module.bias.data.fill_(0.01)
    elif isinstance(module, nn.TransformerEncoderLayer):
        nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
        nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
        nn.init.xavier_uniform_(module.linear1.weight)
        nn.init.xavier_uniform_(module.linear2.weight)
        if module.linear1.bias is not None:
            module.linear1.bias.data.fill_(0.01)
        if module.linear2.bias is not None:
            module.linear2.bias.data.fill_(0.01)


# Feature corruption function
def feature_corruption(x, corruption_ratio=0.3):
    set_seed(42)  # 让 bernoulli 生成固定的随机掩码
    corruption_mask = torch.bernoulli(torch.full(x.shape, 1 - corruption_ratio)).to(x.device)
    return x * corruption_mask


# Encoder network with a three-layer transformer
class Encoder(nn.Module):
    def __init__(self, feature_size, num_heads=2):
        super(Encoder, self).__init__()
        self.transformer_layers = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads,dropout=0.0),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads, dropout=0.0),
            nn.TransformerEncoderLayer(d_model=feature_size, nhead=num_heads,dropout=0.0)
        )

    def forward(self, x):
        # Since Transformer expects seq_length x batch x features, we assume x is already shaped correctly
        return self.transformer_layers(x)


# # Projector network
# class Projector(nn.Module):
#     def __init__(self, feature_size,projector_size):
#         super(Projector, self).__init__()
#         self.linear = nn.Linear(feature_size, projector_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         return self.sigmoid(self.linear(x))


# # Decoder network
# class Decoder(nn.Module):
#     def __init__(self, input_feature_size, output_feature_size):
#         super(Decoder, self).__init__()
#         self.linear = nn.Linear(input_feature_size, output_feature_size)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         return self.sigmoid(self.linear(x))

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

class Decoder(nn.Module):
    def __init__(self, input_feature_size, output_feature_size, hidden_size, dropout_prob=0.2):
        """
        :param input_feature_size: 输入特征维度
        :param output_feature_size: 输出特征维度
        :param hidden_size: 隐藏层维度（可根据需要调整）
        :param dropout_prob: dropout 概率
        """
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_prob),
            
            nn.Linear(hidden_size, output_feature_size),
            nn.Sigmoid()  
        )
        
    def forward(self, x):
        return self.net(x)


def _preprocess_input_data(input_data, train_json):
    set_seed(42)
    cont_data = preprocess_cont_data(
        input_data=input_data[train_json.get("data")["cont_cols"]].copy(),
        file_path=os.path.join("models/standard_scaler.pkl"),
        is_train=True
    )
    set_seed(42)
    cate_data = preprocess_cate_backward_data(
        input_data=input_data[train_json.get("data")["cate_cols"]].copy(),
        file_path=os.path.join("models/backward_encoders.pkl"),
        is_train=True
    )
    return cont_data, cate_data


def _gen_dsnet_dataset(train_datas, target_col, train_json):
    x_cont_datas = []
    x_cate_datas = []
    x_cate_common_datas = []
    common_columns = set()
    for input_data_temp in train_datas:
        x_cont_train, x_cate_train = _preprocess_input_data(input_data_temp, train_json)
        if not common_columns:
            common_columns = set(x_cate_train.columns)
        x_cont_datas.append(x_cont_train)
        x_cate_datas.append(x_cate_train)
        common_columns &= set(x_cate_train.columns)
    common_columns = sorted(list(common_columns))
    with open("common_columns.json", 'w') as file:
        json.dump(common_columns, file)
    for x_cate_data in x_cate_datas:
        x_cate_common_datas.append(x_cate_data[common_columns])
    train_sets = []

    for cont_data, cate_data, train_data in zip(x_cont_datas, x_cate_common_datas, train_datas):
        if len(cate_data.columns):
            if (cont_data.shape[1] + cate_data.shape[1]) % 2:
                cate_data = np.delete(cate_data, 0, axis=1)
            cont_data = preprocess_minmax_data(cont_data)
            cate_data = preprocess_minmax_data(cate_data)
            train_set = DSNETDataset(cont_data, cate_data,
                                     train_data[target_col])
            train_sets.append(train_set)
    if not train_sets:
        raise ValueError("select type has too many classes, please choose other select")
    feature_size = train_sets[0].x_cont.shape[1] + train_sets[0].x_cate.shape[1]
    return train_sets, feature_size


def save_model(model, file_path):
    # 保存模型的状态字典
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# def cross_entropy_loss(y_true, y_pred):
#     if y_true.is_cuda:
#         y_true = y_true.cpu()
#     if y_pred.is_cuda:
#         y_pred = y_pred.cpu()

#     # Detach tensors from the computation graph and convert to numpy
#     y_true = y_true.detach().numpy()
#     y_pred = y_pred.detach().numpy()

#     # Clip predictions to prevent log(0)
#     epsilon = 1e-15
#     y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    
#     # Compute the cross-entropy loss
#     loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    
#     return loss

def pre_train_model(train_sets, batch_size, feature_size, config):
    dataloaders = [ DataLoader(
                        train_set,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0,
                        worker_init_fn=lambda worker_id: np.random.seed(42 + worker_id),
                        generator=torch.Generator().manual_seed(42)  # 确保批次数据一致
                    ) for
                   train_set in train_sets]
    hidden_size=config['model']['hidden_size']
    epochs_Pre_training = config['model']['epochs_Pre_training']
    alpha = config.get("model")["alpha"]
    corruption_ratio = config.get("model")["corruption_ratio"]
    projector_size = config.get("model")["projector_size"]
    # Initialize the components with the feature size
    global s1_salient, s2_salient, m1_mutual, m2_mutual
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # 模型初始化并移动到设备
    pre_encoder = Encoder(feature_size).to(device)
    pre_encoder.apply(init_weights)
    pm_mutual = Projector(feature_size,projector_size,hidden_size).to(device)
    ps_salient = Projector(feature_size,projector_size,hidden_size).to(device)
    d_decoder = Decoder(projector_size, feature_size,hidden_size).to(device)
    label_pretrainer = LabelPretrainer(feature_size, feature_size, 1).to(device)
    feature_attention = FeatureAttention(projector_size).to(device)
    # Loss function and optimizer
    mse_loss = nn.MSELoss()
    # Optimizer for pre-training
    pretrain_optimizer = RMSprop(list(pre_encoder.parameters()) + list(pm_mutual.parameters()) +
                                 list(ps_salient.parameters()) + list(d_decoder.parameters()) , lr=0.0003)

    # Pre-training loop
    print_interval = 50
    best_loss = float('inf')

    for epoch in range(epochs_Pre_training):
        for x1_batch, x2_batch in zip(dataloaders[0], dataloaders[1]):
            # 在训练和推理时，将数据移动到 GPU
            x1_batch = [item.to(device) for item in x1_batch]  # 将 batch 移到 GPU
            x2_batch = [item.to(device) for item in x2_batch]
            # Feature corruption
            x1_corrupted = feature_corruption(x1_batch[0], corruption_ratio).to(dtype=torch.float32)
            x2_corrupted = feature_corruption(x2_batch[0], corruption_ratio).to(dtype=torch.float32)

            # Data encoding
            z1_encoded = pre_encoder(x1_corrupted)
            z2_encoded = pre_encoder(x2_corrupted)
            # Label pretrainer
            pred_x1, pred_x2 = label_pretrainer(z1_encoded, z2_encoded)
            if pred_x1.shape[1] > 1 and pred_x2.shape[1] > 1:
                pred_x1 = torch.argmax(pred_x1, dim=1)
                pred_x2 = torch.argmax(pred_x2, dim=1)
            else:
                pred_x1 = pred_x1.squeeze(1)
                pred_x2 = pred_x2.squeeze(1)

            label_loss = mse_loss(
                x1_batch[1].to(torch.float32),
                pred_x1.to(torch.float32)
            ) + mse_loss(
                x2_batch[1].to(torch.float32),
                pred_x2.to(torch.float32)
            )
            #for classification tasks
            # label_loss = cross_entropy_loss(
            #     x1_batch[1].to(torch.float32),
            #     pred_x1.to(torch.float32)
            # ) + cross_entropy_loss(
            #     x2_batch[1].to(torch.float32),
            #     pred_x2.to(torch.float32)
            # )

            # Feature decoupling
            s1_salient = ps_salient(z1_encoded)
            m1_mutual = pm_mutual(z1_encoded)
            s2_salient = ps_salient(z2_encoded)
            m2_mutual = pm_mutual(z2_encoded)
            # Data reconstruction
            x1_reconstructed = d_decoder(feature_attention(m1_mutual, s1_salient))
            x2_reconstructed = d_decoder(feature_attention(m2_mutual, s2_salient))
            x1_switched = d_decoder(feature_attention(m2_mutual, s1_salient))
            x2_switched = d_decoder(feature_attention(m1_mutual, s2_salient))

            # Calculate loss
            loss = mse_loss(x1_batch[0], x1_reconstructed) + mse_loss(x2_batch[0], x2_reconstructed) + mse_loss(
                x1_batch[0], x1_switched) + mse_loss(x2_batch[0], x2_switched)

            total_loss = loss + alpha * label_loss

            # Update model parameters
            pretrain_optimizer.zero_grad()
            total_loss.backward()
            pretrain_optimizer.step()

        # Print loss every print1_interval epochs
        # if (epoch + 1) % print_interval == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs_Pre_training}], Pre-training Loss: {total_loss.item():.4f}')
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            pre_encoder_path = os.path.join("models/pre_encoder.pth")
            print(f"New best loss: {best_loss:.4f}，Pre-training Epoch: {epoch + 1}")
            save_model(pre_encoder, pre_encoder_path)
            m_projector_path = os.path.join("models/m_projector.pth")
            save_model(pm_mutual, m_projector_path)
            s_projector_path = os.path.join("models/s_projector.pth")
            save_model(ps_salient, s_projector_path)
    # Save s1,s2,m1,m2
    s1_salient_path = os.path.join("models/s1_salient.pt")
    s2_salient_path = os.path.join("models/s2_salient.pt")
    m1_mutual_path = os.path.join("models/m1_mutual.pt")
    m2_mutual_path = os.path.join("models/m2_mutual.pt")

    torch.save(s1_salient, s1_salient_path)
    torch.save(s2_salient, s2_salient_path)
    torch.save(m1_mutual, m1_mutual_path)
    torch.save(m2_mutual, m2_mutual_path)




if __name__ == "__main__":
    with open('train.json', 'r') as f:
        config = json.load(f)

    train_data = pd.read_csv(config.get("data")["train_data"])
    select_type = config.get("data")["select_type"]
    target_col = config.get("data")["target_col"]
    batch_size = config.get("dataloader")["batch_size"]

    select_unique = train_data[select_type].unique()
    train_datas = []

    for select in select_unique:
        train_data_temp = train_data[train_data[select_type] == select]
        train_datas.append(train_data_temp)
    set_seed(42)
    train_sets, feature_size = _gen_dsnet_dataset(train_datas, target_col, config)

    pre_train_model(train_sets, batch_size, feature_size, config)
