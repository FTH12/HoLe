import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn


def preprocess_graph(
    adj: sp.csr_matrix,
    layer: int,
    norm: str = "sym",
    renorm: bool = True,
) -> torch.Tensor:
    """通用的拉普拉斯平滑滤波器。

    Args:
        adj (sp.csr_matrix): 2D稀疏邻接矩阵。
        layer (int): 线性层的数量。
        norm (str): 拉普拉斯矩阵的归一化方式。可以为"sym"或"left"。
        renorm (bool): 是否使用renormalization trick。

    Returns:
        adjs (sp.csr_matrix): 拉普拉斯平滑滤波器。
    """
    adj = sp.coo_matrix(adj)  # 将邻接矩阵转为COO格式
    ident = sp.eye(adj.shape[0])  # 生成单位矩阵

    # 根据renorm参数决定是否对邻接矩阵进行加单位矩阵的处理
    if renorm:
        adj_ = adj + ident
    else:
        adj_ = adj

    rowsum = np.array(adj_.sum(1))  # 计算每行的和（度矩阵）

    # 根据norm参数选择对称归一化或左归一化
    if norm == "sym":
        # 对称归一化
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = (adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo())
        laplacian = ident - adj_normalized  # 拉普拉斯矩阵
    elif norm == "left":
        # 左归一化
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.0).flatten())
        adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()
        laplacian = ident - adj_normalized

    # 为每一层应用拉普拉斯平滑，系数为2/3
    reg = [2 / 3] * layer
    adjs = []
    for i in reg:
        adjs.append(ident - (i * laplacian))  # 存储每层的平滑后的矩阵
    return adjs


def scale(z):
    """特征缩放。

    Args:
        z (torch.Tensor): 隐藏层嵌入。

    Returns:
        z_scaled (torch.Tensor): 缩放后的嵌入。
    """
    zmax = z.max(dim=1, keepdim=True)[0]  # 每行的最大值
    zmin = z.min(dim=1, keepdim=True)[0]  # 每行的最小值
    z_std = (z - zmin) / (zmax - zmin)  # 标准化
    z_scaled = z_std
    return z_scaled  # 返回缩放后的嵌入


class LinTrans(nn.Module):
    """线性变换模型。

    Args:
        layers (int): 线性层的数量。
        dims (list): 每层的隐藏单元数量。
    """

    def __init__(self, layers, dims):
        super().__init__()
        self.layers = nn.ModuleList()  # 创建一个ModuleList来存储多个线性层
        for i in range(layers):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))  # 逐层添加线性层

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入的特征嵌入。

        Returns:
            out (torch.Tensor): 经过隐藏层的输出嵌入。
        """
        out = x
        for layer in self.layers:
            out = layer(out)  # 通过每一层的线性变换
        out = scale(out)  # 对输出进行缩放
        out = F.normalize(out)  # 对结果进行归一化
        return out  # 返回最终嵌入
