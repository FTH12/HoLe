import csv
import os
import random
import time
from pathlib import Path, PurePath  # 用于处理文件和目录路径的标准库
from typing import Tuple  # 用于函数参数的类型提示

import dgl  # Deep Graph Library，用于图神经网络的库
import numpy as np  # 用于数值计算的库
import scipy.sparse as sp  # 用于处理稀疏矩阵的库
import torch  # PyTorch深度学习框架
import torch.nn.functional as F  # PyTorch中的常用神经网络函数
from sklearn.cluster import KMeans, SpectralClustering  # scikit-learn中的聚类算法

# sk_clustering: 使用scikit-learn进行聚类的函数
def sk_clustering(X: torch.Tensor, n_clusters: int, name: str = "kmeans") -> np.ndarray:
    """使用scikit-learn进行聚类。

    Args:
        X (torch.Tensor): 数据嵌入。
        n_clusters (int): 聚类的数量。
        name (str, optional): 聚类算法名称，默认是'kmeans'。

    Raises:
        NotImplementedError: 如果指定的聚类方法未实现。

    Returns:
        np.ndarray: 聚类结果，返回聚类标签。
    """
    if name == "kmeans":  # 如果选择了KMeans聚类
        model = KMeans(n_clusters=n_clusters)  # 初始化KMeans模型
        label_pred = model.fit(X).labels_  # 进行聚类，并获取聚类标签
        return label_pred  # 返回聚类标签
    if name == "spectral":  # 如果选择了谱聚类
        model = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")  # 初始化谱聚类模型
        label_pred = model.fit(X).labels_  # 进行聚类，并获取聚类标签
        return label_pred  # 返回聚类标签
    raise NotImplementedError  # 如果指定的聚类算法未实现，则抛出异常

# make_parent_dirs: 创建指定路径的所有父级目录
def make_parent_dirs(target_path: PurePath) -> None:
    """创建目标路径的所有父目录。

    Args:
        target_path (PurePath): 目标路径。
    """
    if not target_path.parent.exists():  # 如果父级目录不存在
        target_path.parent.mkdir(parents=True, exist_ok=True)  # 递归创建父目录

# refresh_file: 清空目标文件，重新创建
def refresh_file(target_path: str = None) -> None:
    """清空目标路径的文件，如果存在则删除并重新创建文件。

    Args:
        target_path (str): 文件路径。
    """
    if target_path is not None:
        target_path: PurePath = Path(target_path)  # 转换为PurePath对象
        if target_path.exists():  # 如果文件存在
            target_path.unlink()  # 删除文件
        make_parent_dirs(target_path)  # 确保父目录存在
        target_path.touch()  # 创建新的空文件

# save_model: 保存模型、优化器状态、当前训练轮数和损失值到指定文件
def save_model(model_filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, current_epoch: int, loss: float) -> None:
    """保存模型、优化器、当前训练轮数和损失值到文件。

    Args:
        model_filename (str): 保存模型的文件名。
        model (torch.nn.Module): 需要保存的模型。
        optimizer (torch.optim.Optimizer): 优化器对象。
        current_epoch (int): 当前的训练轮数。
        loss (float): 当前的损失值。
    """
    model_path = get_modelfile_path(model_filename)  # 获取模型文件的路径
    torch.save(  # 使用torch.save函数保存模型状态、优化器状态、当前训练轮数和损失值
        {
            "epoch": current_epoch,  # 保存当前的训练轮数
            "model_state_dict": model.state_dict(),  # 保存模型的参数
            "optimizer_state_dict": optimizer.state_dict(),  # 保存优化器的参数
            "loss": loss,  # 保存当前的损失值
        },
        model_path,  # 保存到指定路径
    )

# load_model: 从文件中加载模型、优化器、当前轮数和损失值
def load_model(model_filename: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None, device: torch.device = torch.device("cpu")) -> Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]:
    """从文件中加载模型、优化器、当前训练轮数和损失值。

    Args:
        model_filename (str): 模型文件名。
        model (torch.nn.Module): 模型对象。
        optimizer (torch.optim.Optimizer, optional): 优化器对象。
        device (torch.device, optional): 使用的设备，默认为CPU。

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, int, float]: 返回加载的模型、优化器、当前轮数和损失值。
    """
    model_path = get_modelfile_path(model_filename)  # 获取模型文件的路径
    checkpoint = torch.load(model_path, map_location=device)  # 从文件中加载检查点数据

    model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型的状态
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 加载优化器的状态
    epoch = checkpoint["epoch"]  # 加载训练轮数
    loss = checkpoint["loss"]  # 加载损失值

    return model, optimizer, epoch, loss  # 返回模型、优化器、训练轮数和损失值

# get_modelfile_path: 获取模型文件的完整路径
def get_modelfile_path(model_filename: str) -> PurePath:
    """获取模型文件的路径。

    Args:
        model_filename (str): 模型文件名。

    Returns:
        PurePath: 返回模型文件的路径。
    """
    model_path: PurePath = Path(f".checkpoints/{model_filename}.pt")  # 将模型保存在`.checkpoints`目录下
    if not model_path.parent.exists():  # 如果父目录不存在
        model_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
    return model_path  # 返回模型文件路径

# check_modelfile_exists: 检查模型文件是否存在
def check_modelfile_exists(model_filename: str) -> bool:
    """检查模型文件是否存在。

    Args:
        model_filename (str): 模型文件名。

    Returns:
        bool: 如果文件存在，返回True，否则返回False。
    """
    return get_modelfile_path(model_filename).exists()  # 检查文件是否存在

# get_str_time: 获取当前时间字符串
def get_str_time():
    """获取当前时间的字符串格式，用于文件命名。

    Returns:
        str: 返回当前时间的字符串表示。
    """
    return "time_" + time.strftime("%m%d%H%M%S", time.localtime())  # 按照月日时分秒格式返回时间

# node_homo: 计算节点的同质性
def node_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    """计算节点的同质性。

    Args:
        adj (sp.spmatrix): 图的邻接矩阵。
        labels (torch.Tensor): 节点的标签。

    Returns:
        float: 返回节点的同质性分数。
    """
    adj_coo = adj.tocoo()  # 将邻接矩阵转换为COO格式
    adj_coo.data = ((labels[adj_coo.col] == labels[adj_coo.row]).cpu().numpy().astype(int))  # 检查每条边是否连接了相同标签的节点
    return (np.asarray(adj_coo.sum(1)).flatten() / np.asarray(adj.sum(1)).flatten()).mean()  # 计算平均同质性分数

# edge_homo: 计算边的同质性
def edge_homo(adj: sp.spmatrix, labels: torch.Tensor) -> float:
    """计算边的同质性。

    Args:
        adj (sp.spmatrix): 邻接矩阵。
        labels (torch.Tensor): 节点标签。

    Returns:
        float: 返回边的同质性分数。
    """
    return ((labels[adj.tocoo().col] == labels[adj.tocoo().row]).cpu().numpy() * adj.data).sum() / adj.sum()  # 计算连接相同标签节点的边的比例

# eliminate_zeros: 移除稀疏矩阵中的零值和自环
def eliminate_zeros(adj: sp.spmatrix) -> sp.spmatrix:
    """移除稀疏矩阵中的零值和自环。

    Args:
        adj (sp.spmatrix): 稀疏矩阵。

    Returns:
        sp.spmatrix: 清理后的邻接矩阵。
    """
    adj = adj - sp.dia_matrix(  # 移除对角线上的自环
        (adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)  # 将自环转换为零值
    adj.eliminate_zeros()  # 移除所有零值
    return adj  # 返回处理后的邻接矩阵

# csv2file: 将数据保存为CSV文件
def csv2file(target_path: str, thead: Tuple[str] = None, tbody: Tuple = None, refresh: bool = False, is_dict: bool = False) -> None:
    """将数据保存为CSV文件。

    Args:
        target_path (str): 保存文件的路径。
        thead (Tuple[str], optional): CSV表头，默认为None。
        tbody (Tuple, optional): CSV内容，默认为None。
        refresh (bool, optional): 是否清空文件后再写入，默认为False。
        is_dict (bool, optional): 内容是否为字典格式，默认为False。
    """
    target_path: PurePath = Path(target_path)  # 将目标路径转换为PurePath对象
    if refresh:  # 如果需要清空文件
        refresh_file(target_path)  # 清空文件

    make_parent_dirs(target_path)  # 确保父目录存在

    with open(target_path, "a+", newline="", encoding="utf-8") as csvfile:  # 以追加模式打开文件
        csv_write = csv.writer(csvfile)  # 创建CSV写入器
        if os.stat(target_path).st_size == 0 and thead is not None:  # 如果文件为空且有表头
            csv_write.writerow(thead)  # 写入表头
        if tbody is not None:  # 如果有内容需要写入
            if is_dict:  # 如果内容是字典格式
                dict_writer = csv.DictWriter(csvfile, fieldnames=tbody[0].keys())  # 创建字典写入器
                for elem in tbody:
                    dict_writer.writerow(elem)  # 逐行写入字典内容
            else:
                csv_write.writerow(tbody)  # 以行的形式写入内容

# set_seed: 设置随机种子
def set_seed(seed: int = 4096) -> None:
    """设置随机种子，以确保结果的可重复性。

    注意：DGL库中的卷积和采样器在某些情况下可能是非确定性的。

    Args:
        seed (int, optional): 随机种子，默认为4096。
    """
    if seed is not False:
        os.environ["PYTHONHASHSEED"] = str(seed)  # 设置Python的随机种子
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 设置CUDA的配置以确保可重复性
        random.seed(seed)  # 设置Python的随机数生成器种子
        np.random.seed(seed)  # 设置NumPy的随机数生成器种子
        torch.manual_seed(seed)  # 设置PyTorch的CPU随机数生成器种子
        torch.cuda.manual_seed(seed)  # 设置PyTorch的GPU随机数生成器种子
        torch.cuda.manual_seed_all(seed)  # 如果有多个GPU，设置所有GPU的种子
        torch.backends.cudnn.benchmark = False  # 禁用CuDNN的benchmark模式
        torch.backends.cudnn.deterministic = True  # 启用CuDNN的确定性模式

# set_device: 设置使用的设备（CPU或GPU）
def set_device(gpu: str = "0") -> torch.device:
    """设置PyTorch的设备为CPU或GPU。

    Args:
        gpu (str): 指定使用的GPU编号，默认为'0'。

    Returns:
        torch.device: 返回torch设备对象，可以是GPU或CPU。
    """
    max_device = torch.cuda.device_count() - 1  # 获取最大可用的GPU编号
    if gpu == "none":  # 如果指定为不使用GPU
        print("Use CPU.")
        device = torch.device("cpu")  # 设置为CPU设备
    elif torch.cuda.is_available():  # 如果GPU可用
        if not gpu.isnumeric():
            raise ValueError(f"args.gpu:{gpu} 不是有效的GPU编号。")
        if int(gpu) <= max_device:
            print(f"使用cuda:{gpu}。")  # 输出使用的GPU编号
            device = torch.device(f"cuda:{gpu}")  # 设置为指定GPU设备
            torch.cuda.set_device(device)  # 设置当前设备
        else:
            print(f"cuda:{gpu} 超出了可用设备范围。使用CPU。")
            device = torch.device("cpu")  # 超出范围时使用CPU
    else:
        print("GPU不可用，使用CPU。")
        device = torch.device("cpu")  # 如果没有GPU，使用CPU
    return device  # 返回设备对象

# sparse_mx_to_torch_sparse_tensor: 将Scipy稀疏矩阵转换为PyTorch稀疏张量
def sparse_mx_to_torch_sparse_tensor(sparse_mx: sp.spmatrix) -> torch.Tensor:
    """将Scipy稀疏矩阵转换为PyTorch稀疏张量。

    Args:
        sparse_mx (sp.spmatrix): Scipy稀疏矩阵。

    Returns:
        torch.Tensor: PyTorch稀疏张量。
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)  # 将稀疏矩阵转换为COO格式，并转换为32位浮点数
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # 获取非零元素的索引
    values = torch.from_numpy(sparse_mx.data)  # 获取非零元素的值
    shape = torch.Size(sparse_mx.shape)  # 获取矩阵的形状
    return torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)  # 返回稀疏张量

# torch_sparse_to_dgl_graph: 将PyTorch稀疏张量转换为DGL图
def torch_sparse_to_dgl_graph(torch_sparse_mx):
    """将PyTorch稀疏张量转换为DGL图。

    Args:
        torch_sparse_mx (torch.Tensor): PyTorch稀疏张量。

    Returns:
        dgl.graph: 返回DGL图对象。
    """
    torch_sparse_mx = torch_sparse_mx.coalesce()  # 将稀疏张量转换为紧凑形式
    indices = torch_sparse_mx.indices()  # 获取非零元素的索引
    values = torch_sparse_mx.values()  # 获取非零元素的值
    rows_, cols_ = indices[0, :], indices[1, :]  # 行列索引
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0])  # 创建DGL图
    dgl_graph.edata["w"] = values.detach()  # 将权重值赋给边
    return dgl_graph  # 返回DGL图对象
