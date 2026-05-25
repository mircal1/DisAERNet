import numpy as np
import matplotlib.pyplot as plt
import torch
import json
from sklearn.manifold import TSNE

def visualization():
    # 模拟生成数据
    data_pt_m1 = torch.load("./models/m1_mutual.pt")
    data_pt_m2 = torch.load("./models/m2_mutual.pt")
    data_pt_s1 = torch.load("./models/s1_salient.pt")
    data_pt_s2 = torch.load("./models/s2_salient.pt")
    device = torch.device('cpu')
    data_m1 = data_pt_m1.to(device).detach().numpy()
    data_m2 = data_pt_m2.to(device).detach().numpy()
    data_s1 = data_pt_s1.to(device).detach().numpy()
    data_s2 = data_pt_s2.to(device).detach().numpy()

    # 合并数据
    data = np.concatenate([data_m1, data_m2, data_s1, data_s2], axis=0)
    #labels = ['m1'] * data_m1.shape[0] + ['m2'] * data_m2.shape[0] + ['s1'] * data_s1.shape[0] + ['s2'] * data_s2.shape[0]
    labels = ['Surface_C1'] * data_m1.shape[0] + ['Surface_C2'] * data_m2.shape[0] + ['Atmos_C1'] * data_s1.shape[0] + ['Atmos_C2'] * data_s2.shape[0]

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    # 创建颜色映射
    #colors = {'m1': 'red', 'm2': 'blue', 's1': 'green', 's2': 'orange'}
    colors = {'Surface_C1': 'red', 'Surface_C2': 'blue', 'Atmos_C1': 'green', 'Atmos_C2': 'orange'}

    # 绘制t-SNE图
    plt.figure(figsize=(8, 6))
    for label in np.unique(labels):
        idx = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(data_2d[idx, 0], data_2d[idx, 1], c=colors[label], label=label, alpha=0.7)

    # 图形美化
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(title="Classes", fontsize=10)
    plt.title("t-SNE Visualization", fontsize=14)
    plt.tight_layout()

    # 保存并显示图形
    plt.savefig('pretrain_visualization.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    with open('train.json', 'r') as f:
        config = json.load(f)
    visualization()
