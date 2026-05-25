import json
import os
import random

import pandas as pd
import torch
import torchmetrics
import numpy as np
import inference

from torch import nn
from torch.optim import Adam
from dataset.dsnet import DSNETDataset

from preprocessing import preprocess_cont_data, preprocess_cate_backward_data, \
    preprocess_minmax_data

pd.set_option('future.no_silent_downcasting', True)

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

def load_model(model, file_path, device):
    # 加载模型的状态字典
    model.load_state_dict(torch.load(file_path, map_location=torch.device(device)))
    #print(f"Model loaded from {file_path}")
    return model


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


def _gen_fine_tune_dataset(valid_data, common_values, train_json):
    x_cont_valid, x_cate_valid = _preprocess_input_data(valid_data, train_json)
    x_cate_common_valid = x_cate_valid[common_values]

    if (x_cont_valid.shape[1] + x_cate_common_valid.shape[1]) % 2:
        x_cate_common_valid = np.delete(x_cate_common_valid, 0, axis=1)
    cont_data = preprocess_minmax_data(x_cont_valid)
    cate_data = preprocess_minmax_data(x_cate_common_valid)
    valid_set = DSNETDataset(cont_data, cate_data,
                            valid_data[train_json.get("data")["target_col"]])

    feature_size = valid_set.x_cont.shape[1] + valid_set.x_cate.shape[1]
    return valid_set, feature_size


def save_model(model, file_path):
    # 保存模型的状态字典
    torch.save(model.state_dict(), file_path)
    #print(f"Model saved to {file_path}")

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

def fine_tune_model(valid_set, batch_size, feature_size, num_classes, config):

    valid_dataloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    epochs_Fine_tuning = config['model']['epochs_Fine_tuning']
    lr = config.get("model")["optimizer"]["lr"]
    # Initialize the components with the feature size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    with open("inference.json", "r") as f:
        inference_json = json.load(f)
    test_data = pd.read_csv(inference_json.get("data")["inference"])
    test_set, feature_size = _gen_test_dataset(test_data, common_columns, inference_json) 
    model_config = config.get("model")["net"] 
    set_seed(42)
    # 模型初始化并移动到设备
    pred_predictor = Predictor(feature_size, feature_size//4, num_classes,config['model']['net']).to(device)
    # Pre-training loop
    print_interval = 5

    loaded_pre_encoder = Encoder(feature_size=feature_size).to(device)

    f_encoder = load_model(loaded_pre_encoder, "models/pre_encoder.pth", device)
    best_r2 = -float("inf")
    metric_r2 = torchmetrics.R2Score()
    # Fine-tuning loop
    fine_tuning_loss_function = nn.MSELoss()
    fine_tuning_optimizer = Adam(list(f_encoder.parameters())+list(pred_predictor.parameters()), lr=lr)
    for epoch in range(epochs_Fine_tuning):
        total_loss = 0
        all_predictions = []
        all_labels = []
        for x_batch, labels in valid_dataloader:
            x_batch = x_batch.to(device)
            labels = labels.to(device)
            # Assume that now we have labels
            z_encoded = f_encoder(x_batch)
            predictions = pred_predictor(z_encoded)
            labels = labels.unsqueeze(1)
            all_predictions.append(predictions)
            all_labels.append(labels)
            # Replace 'some_loss_function' with the actual loss function used for fine-tuning
            prediction_loss = fine_tuning_loss_function(predictions, labels)
            total_loss += prediction_loss.item()
            fine_tuning_optimizer.zero_grad()
            prediction_loss.backward()
            fine_tuning_optimizer.step()
        r2_score = metric_r2(torch.cat(all_labels, dim=0), torch.cat(all_predictions, dim=0))

        average_loss = total_loss / len(valid_dataloader)
        # Print loss every print_interval epochs
        # if (epoch + 1) % print_interval == 0:
        #     print(f'Epoch [{epoch + 1}/{epochs_Fine_tuning}], Fine-tuning Loss: {average_loss:.4f}, r2 score: {r2_score:.4f}')
        if r2_score > best_r2 :
            best_r2 = r2_score
            f_encoder_path = os.path.join("models/f_encoder.pth")
            pred_predictor_path = os.path.join("models/pred_predictor.pth")
            save_model(f_encoder, f_encoder_path)
            save_model(pred_predictor, pred_predictor_path)
            print(f'Epoch [{epoch + 1}/{epochs_Fine_tuning}], Fine-tuning Loss: {average_loss:.4f}, r2 score: {r2_score:.4f}')
            predictions, true_labels, inference_loss = inference.inference(test_set, batch_size, feature_size, 1,model_config)




if __name__ == "__main__":
    with open('train.json', 'r') as f:
        config = json.load(f)
    with open("common_columns.json", 'r') as f:
        common_columns = json.load(f)

    valid_data = pd.read_csv(config.get("data")["valid_data"])

    batch_size = config.get("dataloader")["batch_size"]

    valid_set, feature_size = _gen_fine_tune_dataset(valid_data, common_columns, config)

    fine_tune_model(valid_set, batch_size, feature_size, 1, config)
