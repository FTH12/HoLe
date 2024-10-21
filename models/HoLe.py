"""
Homophily-enhanced Structure Learning for Graph Clustering
"""
import copy
import math
import time
from typing import Callable, List

import dgl
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn

from modules import InnerProductDecoder, LinTrans, preprocess_graph, SampleDecoder
from utils import eliminate_zeros, set_seed


class HoLe(nn.Module):
    """HoLe: Homophily-enhanced Structure Learning for Graph Clustering"""

    def __init__(
        self,
        in_feats: int,
        hidden_units: List[int],
        n_clusters: int,
        n_lin_layers: int = 1,
        n_gnn_layers: int = 10,
        n_cls_layers: int = 1,
        lr: float = 0.001,
        n_epochs: int = 400,
        n_pretrain_epochs: int = 400,
        norm: str = "sym",
        renorm: bool = True,
        tb_filename: str = "hole",
        warmup_filename: str = "hole_warmup",
        inner_act: Callable = lambda x: x,
        udp: int = 10,
        reset: bool = False,
        regularization: float = 0,
        seed: int = 4096,
    ):
        """初始化HoLe模型参数

        Args:
            in_feats (int): 输入特征维度
            hidden_units (List[int]): 隐藏层单元数列表
            n_clusters (int): 聚类数量
            n_lin_layers (int, optional): 线性层数，默认为1
            n_gnn_layers (int, optional): GNN层数，默认为10
            n_cls_layers (int, optional): 分类层数，默认为1
            lr (float, optional): 学习率，默认为0.001
            n_epochs (int, optional): 训练轮数，默认为400
            n_pretrain_epochs (int, optional): 预训练轮数，默认为400
            norm (str, optional): 归一化方式，默认为"sym"
            renorm (bool, optional): 是否重新归一化，默认为True
            tb_filename (str, optional): tensorboard文件名
            warmup_filename (str, optional): 预热模型文件名
            inner_act (Callable, optional): 内部激活函数
            udp (int, optional): 更新周期，默认为10
            reset (bool, optional): 是否重置模型，默认为False
            regularization (float, optional): 正则化参数，默认为0
            seed (int, optional): 随机种子，默认为4096
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.n_gnn_layers = n_gnn_layers
        self.n_lin_layers = n_lin_layers
        self.n_cls_layers = n_cls_layers
        self.hidden_units = hidden_units
        self.lr = lr
        self.n_epochs = n_epochs
        self.n_pretrain_epochs = n_pretrain_epochs
        self.norm = norm
        self.renorm = renorm
        self.device = None
        self.sm_fea_s = None
        self.adj_label = None
        self.lbls = None
        self.tb_filename = tb_filename
        self.warmup_filename = warmup_filename
        self.udp = udp
        self.reset = reset
        self.labels = None
        self.adj_sum_raw = None
        self.adj_orig = None
        self.regularization = regularization
        set_seed(seed)

        # 定义输入维度和隐藏层维度
        self.dims = [in_feats] + hidden_units

        # 初始化编码器和聚类层
        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, hidden_units[-1]))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # 初始化优化器和解码器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sample_decoder = SampleDecoder(act=lambda x: x)
        self.inner_product_decoder = InnerProductDecoder(act=inner_act)

        # 保存最佳模型
        self.best_model = copy.deepcopy(self)

    def reset_weights(self):
        """重置模型权重"""
        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def bce_loss(preds, labels, norm=1.0, pos_weight=None):
        """计算二元交叉熵损失"""
        return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    @staticmethod
    def get_fd_loss(z: torch.Tensor, norm: int = 1) -> torch.Tensor:
        """计算特征去相关损失

        Args:
            z (torch.Tensor): 嵌入矩阵
            norm (int, optional): 系数，默认为1

        Returns:
            torch.Tensor: 损失值
        """
        norm_ff = z / (z**2).sum(0, keepdim=True).sqrt()  # 归一化特征
        coef_mat = torch.mm(norm_ff.t(), norm_ff)  # 计算相关矩阵
        coef_mat.div_(2.0)  # 缩放系数矩阵
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)  # 生成索引
        L_fd = norm * F.cross_entropy(coef_mat, a)  # 交叉熵损失
        return L_fd

    @staticmethod
    def target_distribution(q):
        """计算目标分布P

        Args:
            q (torch.Tensor): 软分配矩阵

        Returns:
            torch.Tensor: 目标分布P
        """
        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    def get_cluster_center(self, z=None):
        """计算聚类中心

        Args:
            z (torch.Tensor, optional): 节点嵌入，默认为None

        如果z为None，则从模型中获取嵌入
        """
        if z is None:
            z = self.get_embedding()

        # 使用KMeans聚类算法计算聚类中心
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _ = kmeans.fit_predict(z.data.cpu().numpy())
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(self.device)

    def get_Q(self, z):
        """计算软聚类分配矩阵Q

        Args:
            z (torch.Tensor): 节点嵌入

        Returns:
            torch.Tensor: 软分配矩阵
        """
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def _pretrain(self):
        """预训练模型"""
        best_loss = 1e9  # 初始化最优损失

        for epoch in range(self.n_pretrain_epochs):
            self.train()  # 切换模型为训练模式
            self.optimizer.zero_grad()  # 清除梯度
            t = time.time()

            z = self.encoder(self.sm_fea_s)  # 编码器获取嵌入
            preds = self.inner_product_decoder(z).view(-1)  # 使用解码器生成预测值
            loss = self.bce_loss(preds, self.lbls, norm=self.norm_weights, pos_weight=self.pos_weight)  # 计算损失

            (loss + self.get_fd_loss(z, self.regularization)).backward()  # 计算梯度并反向传播
            self.optimizer.step()  # 更新参数

            cur_loss = loss.item()
            print(f"Cluster Epoch: {epoch}, embeds_loss={cur_loss} time={time.time() - t:.5f}")

            # 保存最佳模型
            if cur_loss < best_loss:
                best_loss = cur_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def _train(self):
        """训练模型"""
        best_loss = 1e9  # 初始化最优损失

        self.get_cluster_center()  # 获取聚类中心
        for epoch in range(self.n_epochs):
            self.train()  # 切换模型为训练模式

            if epoch % self.udp == 0 or epoch == self.n_epochs - 1:
                with torch.no_grad():
                    z_detached = self.get_embedding()  # 获取嵌入
                    Q = self.get_Q(z_detached)  # 计算软分配矩阵Q
                    q = Q.detach().data.cpu().numpy().argmax(1)  # 获取聚类标签

            self.optimizer.zero_grad()  # 清除梯度
            t = time.time()

            z = self.encoder(self.sm_fea_s)  # 编码器获取嵌入
            preds = self.inner_product_decoder(z).view(-1)  # 使用```python
            preds = self.inner_product_decoder(z).view(-1)  # 使用解码器生成预测值
            loss = self.bce_loss(preds, self.lbls, self.norm_weights, self.pos_weight)  # 计算二元交叉熵损失

            q = self.get_Q(z)  # 计算软分配矩阵Q
            p = self.target_distribution(Q.detach())  # 获取目标分布P
            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")  # 计算KL散度损失

            # 损失函数包含三部分：二元交叉熵损失、KL散度损失和正则化损失
            (loss + kl_loss + self.regularization * self.bce_loss(
                self.inner_product_decoder(q).view(-1), preds)).backward()  # 反向传播

            self.optimizer.step()  # 更新模型参数

            cur_loss = loss.item()  # 当前损失

            print(f"Cluster Epoch: {epoch}, embeds_loss={cur_loss:.5f},"
                  f"kl_loss={kl_loss.item()},"
                  f"time={time.time() - t:.5f}")

            # 保存最佳模型
            if cur_loss < best_loss:
                best_loss = cur_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

    def update_adj(
        self,
        adj,
        idx=0,
        ratio=0.2,
        edge_ratio_raw=1,
        del_ratio=0.005,
    ):
        """更新邻接矩阵

        Args:
            adj: 原始邻接矩阵
            idx: 当前迭代轮次
            ratio: 选择的节点比例
            edge_ratio_raw: 边添加比例
            del_ratio: 边删除比例

        Returns:
            更新后的邻接矩阵
        """
        adj_cp = copy.deepcopy(adj).tolil()  # 复制并将邻接矩阵转换为LIL格式
        n_nodes = adj.shape[0]  # 获取节点数

        self.to(self.device)  # 模型转移到设备
        ratio = ratio + 0.01 * idx  # 根据迭代轮次调整比例
        edge_ratio = edge_ratio_raw * (1 + idx)  # 边添加比例调整
        if del_ratio * (1 - 0.05 * idx) > 0:
            del_ratio = del_ratio * (1 - 0.05 * idx)
        else:
            del_ratio = del_ratio * 0.05  # 边删除比例调整

        z_detached = self.get_embedding()  # 获取节点嵌入
        with torch.no_grad():
            soft = self.get_Q(z_detached)  # 计算软分配矩阵Q
        preds = soft.argmax(dim=1).view(-1).cpu().numpy()  # 获取聚类标签

        preds_soft = soft[list(range(adj_cp.shape[0])), preds].cpu().numpy()  # 计算每个节点的软分配值

        top_k_idx = np.array([], dtype="int64")  # 初始化top-k节点索引
        for i in range(self.n_clusters):  # 遍历每个聚类
            c_idx = (preds == i).nonzero()[0]  # 获取属于该聚类的节点索引
            top_k_idx = np.concatenate((
                top_k_idx,
                c_idx[preds_soft[c_idx].argsort()[::-1][:int(len(c_idx) * ratio)]],  # 选择top-k节点
            ))

        top_k_preds = preds[top_k_idx]  # 获取top-k节点的预测标签

        if edge_ratio != 0:  # 添加边
            adj_tensor = torch.IntTensor(self.adj_orig.toarray())  # 将原始邻接矩阵转换为张量
            for c in range(self.n_clusters):  # 遍历每个聚类
                mask = top_k_preds == c  # 获取属于该聚类的节点掩码

                n_nodes_c = mask.sum()  # 获取聚类中节点数量
                cluster_c = top_k_idx[mask]  # 获取聚类中的节点索引
                adj_c = adj_tensor.index_select(
                    0, torch.LongTensor(cluster_c)).index_select(
                        1, torch.LongTensor(cluster_c))  # 获取聚类中的邻接子矩阵

                add_num = math.ceil(edge_ratio * self.adj_sum_raw * n_nodes_c / n_nodes)  # 计算需要添加的边数量
                if add_num == 0:
                    add_num = math.ceil(self.adj_sum_raw * n_nodes_c / n_nodes)

                z_detached = self.get_embedding()  # 获取节点嵌入
                cluster_embeds = z_detached[cluster_c]  # 获取聚类中的嵌入
                sim = torch.matmul(cluster_embeds, cluster_embeds.t())  # 计算节点间相似度

                sim = (
                    sim *
                    (torch.eye(sim.shape[0], dtype=int) ^ 1).to(self.device) +
                    torch.eye(sim.shape[0], dtype=int).to(self.device) * -100)  # 去除自环

                sim = sim * (adj_c ^ 1).to(self.device) + adj_c.to(
                    self.device) * -100  # 排除已存在的边

                sim = sim.view(-1)
                top = sim.sort(descending=True)
                top_sort = top.indices.cpu().numpy()  # 按相似度排序

                sim_top_k = top_sort[:add_num * 2]  # 获取top-k相似对
                xind = sim_top_k // n_nodes_c
                yind = sim_top_k % n_nodes_c
                u = cluster_c[xind]
                v = cluster_c[yind]

                adj_cp[u, v] = adj_cp[v, u] = 1  # 添加边

            print(f"add edges in all: {adj_cp.sum() - adj.sum()}")

        edge_num = adj_cp.sum()  # 更新后的边数量

        del_num = math.ceil(del_ratio * edge_num)  # 计算需要删除的边数量
        if del_num != 0:
            eds = adj_cp.toarray().reshape(-1).astype(bool)  # 将邻接矩阵展平成一维数组

            mask = eds

            negs_inds = np.nonzero(mask)[0]  # 获取非零值的索引

            z_detached = self.get_embedding()  # 获取节点嵌入
            cosine = np.matmul(
                z_detached.cpu().numpy(),
                np.transpose(z_detached.cpu().numpy()),
            ).reshape([-1])  # 计算节点嵌入的余弦相似度
            negs = cosine[mask]  # 获取负边的相似度
            del_num = min(del_num, len(negs))  # 确定要删除的边数

            add_num_actual = adj_cp.sum() - adj.sum()  # 实际添加的边数
            if del_num > add_num_actual:
                print(f"Ooops, too few edges added: {add_num_actual}, but {del_num} needs to be deleted.")
                del_num = int(add_num_actual)

            del_edges = negs_inds[np.argpartition(negs, del_num - 1)[:del_num - 1]]  # 获取要删除的边

            u_del = del_edges // z_detached.shape[0]
            v_del = del_edges % z_detached.shape[0]

            adj_cp[u_del, v_del] = adj_cp[v_del, u_del] = np.array([0] * len(u_del))  # 删除边

            isolated_nodes = (np.asarray(adj_cp.sum(1)).flatten().astype(bool) ^ True).nonzero()[0]  # 获取孤立节点
            if len(isolated_nodes):
                for node_zero in isolated_nodes:
                    with torch.no_grad():
                        max_sim_node_0 = np.argpartition(
                            self.sample_decoder(z_detached[node_zero],
                                                z_detached).cpu().numpy(),
                            n_nodes - 2,
                        )[-2:][0]  # 找到与孤立节点最相似的节点
                        adj_cp[node_zero, max_sim_node_0] = adj_cp[max_sim_node_0, node_zero] = 1  # 为孤立节点添加边

                        neighbors = self.adj_orig[node_zero, :].nonzero()[1]  # 获取原始邻接矩阵中的邻居
                        if len(neighbors):
                            max_sim_node_1 = neighbors[self.sample_decoder(
                                z_detached[node_zero],
                                z_detached[neighbors]).argmax()]  # 找到最相似的邻居
                            adj_cp[node_zero, max_sim_node_1] = adj_cp[max_sim_node_1, node_zero] = 1  # 添加边
        return adj_cp  # 返回更新后的邻接矩阵

    def update_features(self, adj):
        """更新特征矩阵"""
        sm_fea_s = sp.csr_matrix(self.features).toarray()  # 将特征矩阵转换为稀疏格式

        adj_cp = copy.deepcopy(adj)  # 深拷贝邻接矩阵
        self.adj_label = adj_cp  # 设置```python
        self.adj_label = adj_cp  # 设置邻接矩阵标签

        # 预处理图结构，进行规范化操作
        adj_norm_s = preprocess_graph(
            adj_cp,
            self.n_gnn_layers,
            norm=self.norm,
            renorm=self.renorm,
        )

        adj_csr = adj_norm_s[0] if len(adj_norm_s) > 0 else adj_cp  # 获取规范化后的邻接矩阵

        print("Laplacian Smoothing...")
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)  # 进行拉普拉斯平滑处理
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(self.device)  # 将平滑后的特征转为张量

        # 计算正类的权重和规范化权重
        self.pos_weight = torch.FloatTensor([
            (float(adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) /
             adj_csr.sum())
        ]).to(self.device)

        self.norm_weights = (adj_csr.shape[0] * adj_csr.shape[0] / float(
            (adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) * 2))

        # 将邻接矩阵转换为密集格式，并展平为一维张量
        self.lbls = torch.FloatTensor(adj_csr.todense()).view(-1).to(self.device)

    def fit(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        add_edge_ratio=0.2,
        del_edge_ratio=0.1,
        node_ratio=0.2,
        gsl_epochs=10,
        labels=None,
        adj_sum_raw=None,
        load=False,
        dump=True,
    ) -> None:
        """模型训练函数

        Args:
            graph (dgl.DGLGraph): DGL图对象
            device (torch.device): 设备（CPU或GPU）
            add_edge_ratio (float, optional): 边添加比例
            del_edge_ratio (float, optional): 边删除比例
            node_ratio (float, optional): 节点选择比例
            gsl_epochs (int, optional): 结构学习轮数
            labels (torch.Tensor, optional): 节点标签
            adj_sum_raw (int, optional): 原始邻接矩阵边数总和
            load (bool, optional): 是否加载预训练模型
            dump (bool, optional): 是否保存模型
        """
        self.device = device  # 设置设备
        self.features = graph.ndata["feat"]  # 获取图中的节点特征
        adj = self.adj_orig = graph.adj_external(scipy_fmt="csr")  # 获取邻接矩阵
        self.n_nodes = self.features.shape[0]  # 获取节点数量
        self.labels = labels  # 设置节点标签
        self.adj_sum_raw = adj_sum_raw  # 保存原始邻接矩阵边数总和

        adj = eliminate_zeros(adj)  # 移除邻接矩阵中的零值

        self.to(self.device)  # 模型转移到设备

        self.update_features(adj)  # 更新特征

        adj = self.adj_label  # 设置邻接矩阵标签

        # 如果选择加载预训练模型
        from utils.utils import check_modelfile_exists

        if load and check_modelfile_exists(self.warmup_filename):
            from utils.utils import load_model

            self, self.optimizer, _, _ = load_model(
                self.warmup_filename,
                self,
                self.optimizer,
                self.device,
            )

            self.to(self.device)
            print(f"model loaded from {self.warmup_filename} to {self.device}")
        else:
            self._pretrain()  # 进行预训练
            if dump:
                from utils.utils import save_model

                save_model(
                    self.warmup_filename,
                    self,
                    self.optimizer,
                    None,
                    None,
                )
                print(f"dump to {self.warmup_filename}")

        if gsl_epochs != 0:
            self._train()  # 训练模型

            with torch.no_grad():
                z_detached = self.get_embedding()  # 获取嵌入
                Q = self.get_Q(z_detached)  # 计算软分配矩阵Q
                q = Q.detach().data.cpu().numpy().argmax(1)  # 获取聚类标签

        adj_pre = copy.deepcopy(adj)  # 复制邻接矩阵

        for gls_ep in range(1, gsl_epochs + 1):
            print(f"==============GSL epoch:{gls_ep} ===========")

            # 更新邻接矩阵
            adj_new = eliminate_zeros(
                self.update_adj(
                    adj_pre,
                    idx=gls_ep,
                    ratio=node_ratio,
                    del_ratio=del_edge_ratio,
                    edge_ratio_raw=add_edge_ratio,
                ))

            self.update_features(adj=adj_new)  # 更新特征

            self._train()  # 训练模型

            adj_pre = copy.deepcopy(self.adj_label)  # 更新邻接矩阵标签

            if self.reset:
                self.reset_weights()  # 重置模型权重
                self.to(self.device)  # 模型转移到设备

    def get_embedding(self, best=True):
        """获取节点嵌入

        Args:
            best (bool, optional): 是否返回最佳模型的嵌入，默认为True

        Returns:
            torch.Tensor: 节点嵌入
        """
        with torch.no_grad():
            mu = (self.best_model.encoder(self.sm_fea_s) if best else self.encoder(self.sm_fea_s))  # 获取嵌入
        return mu.detach()  # 返回节点嵌入
