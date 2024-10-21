"""
Homophily-enhanced Graph Structure Learning for Graph Clustering
"""
import copy
import gc
import random
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
from utils import eliminate_zeros, set_seed, sparse_mx_to_torch_sparse_tensor, torch_sparse_to_dgl_graph


class HoLe(nn.Module):
    """HoLe模型用于图聚类的同质性增强结构学习"""

    def __init__(self, in_feats: int, hidden_units: List, n_clusters: int, n_lin_layers: int = 1, n_gnn_layers: int = 10,
                 n_cls_layers: int = 1, lr: float = 0.001, n_epochs: int = 400, n_pretrain_epochs: int = 400,
                 norm: str = "sym", renorm: bool = True, tb_filename: str = "hole", warmup_filename: str = "hole_warmup",
                 inner_act: Callable = lambda x: x, udp=10, reset=False, regularization=0, seed: int = 4096):
        """
        初始化HoLe模型参数

        Args:
            in_feats (int): 输入特征维度
            hidden_units (List): 隐藏层单元数列表
            n_clusters (int): 聚类数量
            n_lin_layers (int, optional): 线性层数，默认为1
            n_gnn_layers (int, optional): GNN层数，默认为10
            n_cls_layers (int, optional): 分类层数，默认为1
            lr (float, optional): 学习率，默认为0.001
            n_epochs (int, optional): 训练轮数，默认为400
            n_pretrain_epochs (int, optional): 预训练轮数，默认为400
            norm (str, optional): 归一化方式，默认为'sym'
            renorm (bool, optional): 是否重新归一化，默认为True
            tb_filename (str, optional): tensorboard文件名
            warmup_filename (str, optional): 预热模型文件名
            inner_act (Callable, optional): 内部激活函数
            udp (int, optional): 更新周期
            reset (bool, optional): 是否重置模型
            regularization (float, optional): 正则化参数
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
        self.pseudo_label = None
        self.hard_mask = None
        self.gsl_embedding = 0
        self.tb_filename = tb_filename
        self.warmup_filename = warmup_filename
        self.udp = udp
        self.reset = reset
        self.regularization = regularization
        set_seed(seed)  # 设置随机种子，确保结果可复现

        # 定义特征维度和隐藏层
        self.dims = [in_feats] + hidden_units
        self.encoder = LinTrans(self.n_lin_layers, self.dims)  # 初始化编码器
        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, hidden_units[-1]))  # 聚类层参数
        torch.nn.init.xavier_normal_(self.cluster_layer.data)  # Xavier初始化

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)  # Adam优化器
        self.sample_decoder = SampleDecoder(act=lambda x: x)  # 解码器
        self.inner_product_decoder = InnerProductDecoder(act=inner_act)  # 内积解码器

        self.best_model = copy.deepcopy(self)  # 保存最优模型

    def reset_weights(self):
        """重置模型权重"""
        self.encoder = LinTrans(self.n_lin_layers, self.dims)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def bce_loss(preds, labels, norm=1.0, pos_weight=None):
        """计算二元交叉熵损失"""
        return norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    @staticmethod
    def get_fd_loss(z, norm=1):
        """计算特征去相关损失"""
        norm_ff = z / (z**2).sum(0, keepdim=True).sqrt()  # 归一化嵌入
        coef_mat = torch.mm(norm_ff.t(), norm_ff)  # 计算嵌入相关矩阵
        coef_mat.div_(2.0)  # 缩放矩阵
        a = torch.arange(coef_mat.size(0), device=coef_mat.device)  # 生成索引
        L_fd = norm * F.cross_entropy(coef_mat, a)  # 计算交叉熵损失
        return L_fd

    def get_cluster_center(self):
        """获取聚类中心"""
        z = self.get_embedding()  # 获取节点嵌入
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)  # KMeans聚类
        _ = kmeans.fit_predict(z.data.cpu().numpy())  # 聚类预测
        self.cluster_layer.data = torch.Tensor(kmeans.cluster_centers_).to(self.device)  # 更新聚类中心

    def _re_pretrain(self):
        """重新进行预训练"""
        best_loss = 1e9  # 初始化最优损失

        for epoch in range(self.n_pretrain_epochs):  # 预训练周期
            self.train()
            self.optimizer.zero_grad()  # 梯度清零
            t = time.time()

            z = self.encoder(self.sm_fea_s)  # 获取节点嵌入

            index = list(range(len(z)))
            split_list = data_split(index, 10000)  # 将数据分批处理

            ins_loss = 0  # 初始化损失

            for batch in split_list:  # 逐批训练
                z_batch = z[batch]
                A_batch = torch.FloatTensor(self.lbls[batch, :][:, batch].toarray()).to(self.device)  # 获取邻接矩阵
                preds = self.inner_product_decoder(z_batch).view(-1)  # 解码预测值

                # 计算二元交叉熵损失
                pos_weight_b = torch.FloatTensor([float(A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) / A_batch.sum()]).to(self.device)
                norm_weights_b = (A_batch.shape[0] * A_batch.shape[0] / float((A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) * 2))
                loss = self.bce_loss(preds, A_batch.view(-1), norm_weights_b, pos_weight_b)

                ins_loss += loss.item()

                (loss).backward(retain_graph=True)  # 反向传播
                self.optimizer.step()

            torch.cuda.empty_cache()  # 清除缓存
            gc.collect()

            print(f"Cluster Epoch: {epoch}, ins_loss={ins_loss}, time={time.time() - t:.5f}")

            if ins_loss < best_loss:
                del self.best_model  # 删除之前的最优模型
                self.best_model = copy.deepcopy(self.to(self.device))  # 保存新的最优模型

    def _re_ce_train(self):
        """模型训练过程，基于KL散度与二分类交叉熵损失训练"""
        best_loss = 1e9  # 初始化最优损失为较大值

        self.get_cluster_center()  # 获取当前聚类中心

        for epoch in range(self.n_epochs):  # 迭代训练
            self.train()

            # 定期或在最后一轮计算Q值
            if epoch % self.udp == 0 or epoch == self.n_epochs - 1:
                with torch.no_grad():
                    z_detached = self.get_embedding()  # 获取嵌入
                    Q = self.get_Q(z_detached)  # 计算Q矩阵
                    q = Q.detach().data.cpu().numpy().argmax(1)  # 获取聚类结果

            t = time.time()

            z = self.encoder(self.sm_fea_s)  # 对特征进行编码

            index = list(range(len(z)))  # 获取所有节点的索引
            split_list = data_split(index, 10000)  # 分批次处理数据

            ins_loss = 0  # 初始化批次损失

            for batch in split_list:  # 逐批次计算损失
                z_batch = z[batch]  # 获取批次嵌入
                A_batch = torch.FloatTensor(self.lbls[batch, :][:, batch].toarray()).to(self.device)  # 提取对应的邻接矩阵
                preds = self.inner_product_decoder(z_batch).view(-1)  # 预测结果

                q = self.get_Q(z_batch)  # 计算Q值
                p = target_distribution(Q[batch].detach())  # 计算目标分布
                kl_loss = F.kl_div(q.log(), p, reduction="batchmean")  # 计算KL散度损失

                # 计算二元交叉熵损失
                pos_weight_b = torch.FloatTensor(
                    [(float(A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) / A_batch.sum())]).to(self.device)
                norm_weights_b = (A_batch.shape[0] * A_batch.shape[0] / float(
                    (A_batch.shape[0] * A_batch.shape[0] - A_batch.sum()) * 2))
                loss = self.bce_loss(preds, A_batch.view(-1), norm_weights_b, pos_weight_b)
                ins_loss += loss.item()

                self.optimizer.zero_grad()  # 梯度清零
                (loss + kl_loss).backward(retain_graph=True)  # 反向传播，计算梯度
                self.optimizer.step()  # 更新参数

            print(f"Cluster Epoch: {epoch}, ins_loss={ins_loss}, kl_loss={kl_loss.item()}, time={time.time() - t:.5f}")

            if ins_loss < best_loss:  # 保存当前最优模型
                best_loss = ins_loss
                del self.best_model
                self.best_model = copy.deepcopy(self.to(self.device))

            torch.cuda.empty_cache()  # 清理缓存
            gc.collect()

    def get_pseudo_label(self, node_rate=0.2):
        """生成伪标签，选择部分节点用于监督学习"""
        with torch.no_grad():
            z_detached = self.get_embedding()  # 获取节点嵌入
            Q = self.get_Q(z_detached)  # 计算Q矩阵
            soft = Q.detach()  # 软聚类结果

        hard = soft.argmax(dim=1).view(-1).cpu().numpy()  # 将软聚类结果转为硬标签

        hard_mask = np.array([False for _ in range(len(hard))], dtype=np.bool)  # 初始化硬标签掩码
        for c in range(self.n_clusters):  # 遍历每个聚类
            add_num = int(node_rate * soft.shape[0])  # 计算需要添加的节点数

            c_col = soft[:, c].detach().cpu().numpy()  # 获取属于当前聚类的概率
            c_col_idx = c_col.argsort()[::-1][:add_num]  # 按概率排序取前add_num个节点
            top_c_idx = c_col_idx  # 选中的节点
            hard[top_c_idx] = c  # 将选中的节点赋予聚类标签

            hard_mask[top_c_idx] = True  # 更新硬标签掩码

            print(f"class {c}, num={len(top_c_idx)}")

        hard[~hard_mask] = -1  # 未选中的节点赋值为-1
        self.pseudo_label = hard  # 保存伪标签
        self.hard_mask = hard_mask  # 保存硬标签掩码

    def _train(self, node_ratio=0.2):
        """执行训练过程"""
        self.get_pseudo_label(node_ratio)  # 获取伪标签
        self._re_ce_train()  # 进行训练

    def _pretrain(self):
        """预训练过程"""
        self._re_pretrain()  # 调用重新预训练函数

    def add_edge(self, edges, label, n_nodes, add_edge_rate, z):
        """根据节点相似性增加边"""
        u, v = edges[0].numpy(), edges[1].numpy()  # 获取边的两个端点
        if add_edge_rate == 0:  # 不增加边的情况
            row = u.tolist()
            col = v.tolist()
            data = [1 for _ in range(len(row))]
            adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))  # 创建稀疏矩阵
            adj_csr[adj_csr > 1] = 1  # 确保矩阵中每个元素不大于1
            adj_csr = eliminate_zeros(adj_csr)  # 消除零元素
            print(f"after add edge, final edges num={adj_csr.toarray().sum()}")
            return adj_csr

        lbl_np = label  # 节点标签
        final_u, final_v = [], []  # 初始化新的边列表
        for i in np.unique(lbl_np):  # 遍历每个聚类
            if i == -1:
                continue
            print(f"************ label == {i} ************")
            same_class_bool = np.array(lbl_np == i, dtype=np.bool)  # 选择属于同一聚类的节点
            same_class_idx = np.nonzero(same_class_bool)[0]  # 获取属于该聚类的节点索引
            nodes = same_class_idx

            n_nodes_c = len(nodes)  # 当前聚类中的节点数
            add_num = int(add_edge_rate * n_nodes_c ** 2)  # 计算需要增加的边数量
            cluster_embeds = z[nodes]  # 获取当前聚类节点的嵌入

            sims = []
            CHUNK = 1024 * 4  # 定义批次大小

            CTS = len(cluster_embeds) // CHUNK  # 计算分批次数
            if len(cluster_embeds) % CHUNK != 0:
                CTS += 1
            for j in range(CTS):  # 分批次计算节点之间的相似度
                a = j * CHUNK
                b = (j + 1) * CHUNK
                b = min(b, len(cluster_embeds))

                cts = torch.matmul(cluster_embeds, cluster_embeds[a:b].T).view(-1)  # 计算余弦相似度
                sims.append(cts.cpu())
            sims = torch.cat(sims).cpu()
            sim_top_k = sims.sort(descending=True).indices.numpy()[:add_num]  # 选出相似度最高的节点对
            xind = sim_top_k // n_nodes_c
            yind = sim_top_k % n_nodes_c

            new_u = nodes[xind]  # 新的边的起始节点
            new_v = nodes[yind]  # 新的边的结束节点
            final_u.extend(new_u.tolist())  # 添加新的边
            final_v.extend(new_v.tolist())

        # 构造新的边
        rw = final_u + final_v
        cl = final_v + final_u
        data0 = [1 for _ in range(len(rw))]
        extra_adj = sp.csr_matrix((data0, (rw, cl)), shape=(n_nodes, n_nodes))

        extra_adj[extra_adj > 1] = 1  # 确保邻接矩阵中每个元素不大于1
        extra_adj = eliminate_zeros(extra_adj)

        row = u.tolist() + final_u + final_v
        col = v.tolist() + final_v + final_u
        data = [1 for _ in range(len(row))]
        adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        adj_csr[adj_csr > 1] = 1  # 确保邻接矩阵中每个元素不大于1
        adj_csr = eliminate_zeros(adj_csr)

        print(f"after add edge, final edges num={adj_csr.sum()}")
        return adj_csr

    def del_edge(self, edges, label, n_nodes, del_edge_rate=0.25):
        """删除部分边以提高图的稀疏性"""
        u, v = edges[0].numpy(), edges[1].numpy()  # 提取边的两个节点
        lbl_np = label  # 获取节点标签
        if del_edge_rate == 0:  # 如果删除率为0，则不删除任何边
            row = u.tolist()  # 行索引
            col = v.tolist()  # 列索引
            data = [1 for _ in range(len(row))]  # 数据填充
            adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))  # 构造邻接矩阵

            adj_csr[adj_csr > 1] = 1  # 确保边权不大于1
            adj_csr = eliminate_zeros(adj_csr)  # 消除零元素
            print(f"after del edge, final edges num={adj_csr.sum()}")  # 打印删除后的边数
            return adj_csr

        # 找出不同标签的节点对的边
        inter_class_bool = np.array((lbl_np[u] != -1) & (lbl_np[v] != -1) & (lbl_np[u] != lbl_np[v]), dtype=np.bool)
        inter_class_idx = np.nonzero(inter_class_bool)[0]  # 获取这些边的索引
        inter_class_edge_len = len(inter_class_idx)  # 不同类节点对之间的边数量

        # 获取同类节点对的边索引
        all_other_edges_idx = np.nonzero(~inter_class_bool)[0].tolist()

        np.random.shuffle(inter_class_idx)  # 随机打乱不同类节点对边的顺序

        # 根据删除率计算保留的边数量
        remain_edge_len = inter_class_edge_len - int(inter_class_edge_len * del_edge_rate)
        inter_class_idx = inter_class_idx[:remain_edge_len].tolist()  # 保留剩余的边
        final_edges_idx = all_other_edges_idx + inter_class_idx  # 合并保留的同类和不同类边

        new_u = u[final_edges_idx].tolist()  # 获取新保留的边的起点
        new_v = v[final_edges_idx].tolist()  # 获取新保留的边的终点

        # 构造新的邻接矩阵
        row = new_u
        col = new_v
        data = [1 for _ in range(len(row))]
        adj_csr = sp.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))

        adj_csr[adj_csr > 1] = 1  # 确保邻接矩阵中每个元素不大于1
        adj_csr = eliminate_zeros(adj_csr)  # 消除零元素

        print(f"after del edge, final edges num={adj_csr.sum()}")  # 打印删除后的边数

        return adj_csr  # 返回新的邻接矩阵

    def update_features(self, adj):
        """根据图的邻接矩阵更新节点特征"""
        adj_norm_s = preprocess_graph(adj, self.n_gnn_layers, norm=self.norm, renorm=self.renorm)  # 归一化邻接矩阵
        adj_csr = adj_norm_s[0]  # 取出第一个归一化矩阵
        sm_fea_s = sp.csr_matrix(self.features).toarray()  # 将特征矩阵转换为稀疏矩阵

        print("Laplacian Smoothing...")  # 拉普拉斯平滑
        for a in adj_norm_s:
            sm_fea_s = a.dot(sm_fea_s)  # 对特征矩阵进行平滑
        self.sm_fea_s = torch.FloatTensor(sm_fea_s).to(self.device)  # 转为tensor并放到设备上

        # 计算正样本的权重
        self.pos_weight = torch.FloatTensor(
            [(float(adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) / adj_csr.sum())]).to(self.device)
        # 计算规范化的权重
        self.norm_weights = (
                adj_csr.shape[0] * adj_csr.shape[0] / float((adj_csr.shape[0] * adj_csr.shape[0] - adj_csr.sum()) * 2))

        self.lbls = adj_csr  # 更新标签为当前邻接矩阵

    def fit(
        self,
        graph: dgl.DGLGraph,
        device: torch.device,
        add_edge_ratio=0.04,
        node_ratio=0.2,
        del_edge_ratio=0,
        gsl_epochs=10,
        labels=None,
        adj_sum_raw=None,
        load=False,
        dump=True,
    ):
        """训练模型"""
        self.device = device  # 设置设备
        self.features = graph.ndata["feat"]  # 获取节点特征
        self.labels = labels  # 获取节点标签
        adj = graph.adj_external(scipy_fmt="csr")  # 获取图的邻接矩阵
        edges = graph.edges()  # 获取图中的边
        n_nodes = self.features.shape[0]  # 获取节点数量

        adj = eliminate_zeros(adj)  # 消除邻接矩阵中的零元素
        self.adj = adj  # 保存邻接矩阵
        self.to(self.device)  # 将模型移至设备

        self.update_features(adj)  # 更新节点特征

        # 检查是否存在预训练模型
        from utils.utils import check_modelfile_exists
        if load and check_modelfile_exists(self.warmup_filename):
            from utils.utils import load_model
            self, self.optimizer, _, _ = load_model(self.warmup_filename, self, self.optimizer, self.device)
            self.to(self.device)
            print(f"model loaded from {self.warmup_filename} to {self.device}")
        else:
            self._pretrain()  # 进行预训练
            if dump:
                from utils.utils import save_model
                save_model(self.warmup_filename, self, self.optimizer, None, None)

        self.get_pseudo_label(node_ratio)  # 获取伪标签

        adj_pre = copy.deepcopy(adj)  # 备份邻接矩阵

        for gls_ep in range(gsl_epochs):  # 进行图结构学习
            torch.cuda.empty_cache()  # 清空CUDA缓存
            gc.collect()  # 进行垃圾回收

            print(f"==============GSL epoch:{gls_ep} ===========")
            with torch.no_grad():
                z_detached = self.get_embedding()  # 获取嵌入

            adj_new = self.add_edge(edges, self.pseudo_label, n_nodes, add_edge_ratio, z_detached)  # 增加边

            # 更新图
            graph_new = torch_sparse_to_dgl_graph(sparse_mx_to_torch_sparse_tensor(adj_new))
            graph_new.ndata["feat"] = graph.ndata["feat"]
            graph_new.ndata["label"] = graph.ndata["label"]
            graph_new.ndata["emb"] = z_detached.cpu()
            edges = graph_new.edges()

            adj_new = self.del_edge(edges, self.pseudo_label, n_nodes, del_edge_ratio)  # 删除边

            # 更新图
            graph_new = torch_sparse_to_dgl_graph(sparse_mx_to_torch_sparse_tensor(adj_new))
            graph_new.ndata["feat"] = graph.ndata["feat"]
            graph_new.ndata["label"] = graph.ndata["label"]
            graph_new.ndata["emb"] = z_detached.cpu()
            edges = graph_new.edges()

            self.update_features(adj=adj_new)  # 更新节点特征
            self._train(node_ratio)  # 进行训练

            if self.reset:  # 重置模型权重
                self.reset_weights()
                self.to(self.device)

    def get_embedding(self):
        """获取节点嵌入

        Returns:
            torch.Tensor: 返回经过最佳模型编码器得到的节点嵌入
        """
        with torch.no_grad():  # 禁用梯度计算
            mu = self.best_model.encoder(self.sm_fea_s)  # 通过模型编码器计算节点嵌入
        return mu.detach()  # 返回分离后的嵌入张量

    def get_Q(self, z):
        """获取软聚类分配的概率分布

        Args:
            z (torch.Tensor): 节点嵌入

        Returns:
            torch.Tensor: 节点对应的软聚类概率分布
        """
        # 计算嵌入与聚类中心的距离
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.cpu().unsqueeze(1) - self.cluster_layer.cpu(), 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)  # 计算概率
        q = (q.t() / torch.sum(q, 1)).t()  # 正则化概率分布
        return q  # 返回聚类分配的概率分布

def data_split(full_list, n_sample):
    """将列表分割为指定大小的子列表

    Args:
        full_list (list): 完整列表
        n_sample (int): 每个子列表的大小

    Returns:
        list: 切割后的子列表集合
    """
    offset = n_sample  # 每个子列表的元素数量
    random.shuffle(full_list)  # 随机打乱列表
    len_all = len(full_list)  # 获取列表长度
    index_now = 0  # 当前索引
    split_list = []  # 保存分割后的子列表
    while index_now < len_all:
        if index_now + offset > len_all:  # 如果当前索引加偏移量超出总长度，取剩余部分
            split_list.append(full_list[index_now:len_all])
        else:
            split_list.append(full_list[index_now:index_now + offset])  # 按偏移量切分列表
        index_now += offset  # 更新当前索引
    return split_list  # 返回切割后的子列表

def target_distribution(q):
    """获取目标分布P，用于聚类

    Args:
        q (torch.Tensor): 软聚类分配的概率分布

    Returns:
        torch.Tensor: 目标分布P
    """
    weight = q ** 2 / q.sum(0)  # 计算权重，公式中的q的平方除以其列和
    return (weight.t() / weight.sum(1)).t()  # 归一化权重并返回目标分布
