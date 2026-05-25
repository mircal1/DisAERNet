import pickle

import pandas as pd
import torch
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler,MinMaxScaler
from category_encoders import BackwardDifferenceEncoder
from scipy import sparse


def preprocess_cate_backward_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    if is_train:
        backward_encoder = BackwardDifferenceEncoder(cols=input_data.columns.tolist())

        input_data = backward_encoder.fit_transform(input_data)
        with open(file_path, "wb") as f:
            pickle.dump(backward_encoder, f)
    else:
        with open(file_path, "rb") as f:
            backward_encoder = pickle.load(f)
        input_data = backward_encoder.transform(input_data)
    return input_data


def preprocess_onehot_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    if is_train:
        onehot_encoder = OneHotEncoder(sparse_output=False)
        input_data = onehot_encoder.fit_transform(input_data)
        with open(file_path, "wb") as f:
            pickle.dump(onehot_encoder, f)
    else:
        with open(file_path, "rb") as f:
            onehot_encoder = pickle.load(f)
        input_data = onehot_encoder.transform(input_data)
    return input_data

def preprocess_minmax_data(input_data: pd.DataFrame):
    torch.manual_seed(42)  # 关键，保证初始化一致
    minmax_scaler = MinMaxScaler()
    input_data = minmax_scaler.fit_transform(input_data)
    return input_data

def preprocess_cont_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    if is_train:
        standard_scaler = StandardScaler()
        input_data = standard_scaler.fit_transform(input_data)
        with open(file_path, "wb") as f:
            pickle.dump(standard_scaler, f)
    else:
        with open(file_path, "rb") as f:
            standard_scaler = pickle.load(f)
        input_data = standard_scaler.transform(input_data)
    return input_data


def preprocess_cate_data(input_data: pd.DataFrame, file_path: str, is_train: bool):
    cate_cols = input_data.columns
    if is_train:
        label_encoders = {}
        for c in cate_cols:
            label_encoder = LabelEncoder()
            input_data[c] = label_encoder.fit_transform(input_data[c])
            label_encoders[c] = label_encoder
        with open(file_path, "wb") as f:
            pickle.dump(label_encoders, f)
    else:
        with open(file_path, "rb") as f:
            label_encoders = pickle.load(f)
        for c in cate_cols:
            input_data[c] = label_encoders[c].transform(input_data[c])
    input_data = input_data.values
    return input_data

def process_tablur_geotiff(df_inference: pd.DataFrame, target_col: str, bound: list, resolution: float):
    height, width = int((bound[0] - bound[2])/ resolution), int((bound[3] - bound[1])/ resolution)
    extent = (bound[1],resolution, 0, bound[0], 0, -resolution)
        
    df_inference['col'] = ((df_inference['lon'] - bound[1])/resolution).astype(int)
    df_inference['row'] = ((bound[0] - df_inference['lat'])/resolution).astype(int)
    sparse_arr = sparse.coo_matrix((df_inference.loc[:, f'{target_col}_PRED'],\
                                    (df_inference.loc[:, 'row'],df_inference.loc[:, 'col'])), shape=(height, width))
    dst_arr = sparse_arr.toarray()
    return dst_arr,height,width,extent