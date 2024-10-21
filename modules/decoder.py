"""
Decoder layers
"""
# pylint: disable=invalid-name
import torch
import torch.nn.functional as F
from torch import nn


class InnerProductDecoder(nn.Module):
    """使用内积进行预测的解码器。

    Args:
        dropout (float, optional): dropout率，用于训练时随机丢弃一部分连接。默认为0.0。
        act (function, optional): 激活函数。默认为恒等函数lambda x: x。
    """
    def __init__(self, dropout: float = 0.0, act=lambda x: x):
        super().__init__()
        self.dropout = dropout  # 设置dropout率
        self.act = act  # 设置激活函数

    def forward(self, z):
        """前向传播，基于内积计算节点之间的相似度。

        Args:
            z (torch.Tensor): 输入节点嵌入

        Returns:
            torch.Tensor: 预测的邻接矩阵
        """
        # 对输入的节点嵌入z进行dropout操作（训练时会随机丢弃一部分节点嵌入）
        z = F.dropout(z, self.dropout, training=self.training)
        # 通过矩阵内积计算节点间的相似度，得到邻接矩阵
        adj = self.act(torch.mm(z, z.t()))
        return adj  # 返回激活后的邻接矩阵


class SampleDecoder(nn.Module):
    """解码器模型，使用内积计算节点间相似度。

    Args:
        act (function, optional): 解码器的激活函数。默认为torch.sigmoid。
    """
    def __init__(self, act=torch.sigmoid):
        super().__init__()
        self.act = act  # 设置激活函数

    def forward(self, zx, zy):
        """前向传播，计算两个节点嵌入之间的相似度。

        Args:
            zx (torch.Tensor): x轴上的节点嵌入
            zy (torch.Tensor): y轴上的节点嵌入

        Returns:
            torch.Tensor: 预测的相似度
        """
        # 通过点乘计算两个节点嵌入之间的相似度
        sim = (zx * zy).sum(1)
        # 应用激活函数（默认sigmoid）得到最终的相似度
        sim = self.act(sim)

        return sim  # 返回激活后的相似度
