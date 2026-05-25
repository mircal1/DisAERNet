import torch
from torch.utils.data import Dataset


class DSNETDataset(Dataset):

    def __init__(self, x_cont, x_cate, y=None):
    
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cate = torch.LongTensor(x_cate)
        if y is not None:
            self.y = torch.FloatTensor(y.values)
        else:
            self.y = None

    def __len__(self):
        return len(self.x_cont)
    
    def __getitem__(self, idx):
        item = torch.cat((self.x_cont[idx], self.x_cate[idx]),0)
        # {
        #     "X_CONT": self.x_cont[idx],
        #     "X_CATE": self.x_cate[idx],
        # }
        # if self.y is not None:
        #     item["Y"] = self.y[idx]
        return item, self.y[idx]
