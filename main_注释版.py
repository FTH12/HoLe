# pylint: disable=line-too-long,invalid-name, 引入库和禁用Pylint的一些检查规则
import argparse  # 用于处理命令行参数
import random  # 用于生成随机数
import scipy.sparse as sp  # 处理稀疏矩阵的操作库
import torch  # Pytorch深度学习库
from graph_datasets import load_data  # 引入自定义函数`load_data`，用于加载图数据集
from models import HoLe  # 引入HoLe模型
from models import HoLe_batch  # 引入批量处理的HoLe模型
from utils import check_modelfile_exists  # 引入函数，检查模型文件是否已存在
from utils import csv2file  # 引入函数，将结果保存为CSV文件
from utils import evaluation  # 引入函数，用于聚类的结果评估
from utils import get_str_time  # 引入函数，获取当前时间的字符串格式
from utils import set_device  # 引入函数，设置训练使用的设备（CPU或GPU）

# 主程序入口
if __name__ == "__main__":
    # 创建ArgumentParser对象，解析命令行参数
    parser = argparse.ArgumentParser(
        prog="HoLe",  # 程序名称
        description="Homophily-enhanced Structure Learning for Graph Clustering",  # 程序描述
    )
    # 添加参数：数据集名称，默认Cora
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset used in the experiment",
    )
    # 添加参数：GPU ID，默认0
    parser.add_argument(
        "-g",
        "--gpu_id",
        type=int,
        default=0,
        help="gpu id",
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 初始化一个字典，用于存放最终参数
    final_params = {}

    # 设置一些通用的超参数
    dim = 500  # 嵌入维度
    n_lin_layers = 1  # 线性层的数量
    dump = True  # 是否存储结果
    device = set_device(str(args.gpu_id))  # 设置设备，使用GPU或CPU

    # 各个数据集的学习率
    lr = {
        "Cora": 0.001,
        "Citeseer": 0.001,
        "ACM": 0.001,
        "Pubmed": 0.001,
        "BlogCatalog": 0.001,
        "Flickr": 0.001,
        "Reddit": 2e-5,
    }

    # 不同数据集使用的GNN层数
    n_gnn_layers = {
        "Cora": [8],
        "Citeseer": [3],
        "ACM": [3],
        "Pubmed": [35],
        "BlogCatalog": [1],
        "Flickr": [1],
        "Reddit": [3],
    }

    # 不同数据集的预训练轮数
    pre_epochs = {
        "Cora": [150],
        "Citeseer": [150],
        "ACM": [200],
        "Pubmed": [50],
        "BlogCatalog": [150],
        "Flickr": [300],
        "Reddit": [3],
    }

    # 不同数据集的训练轮数
    epochs = {
        "Cora": 50,
        "Citeseer": 150,
        "ACM": 150,
        "Pubmed": 200,
        "BlogCatalog": 150,
        "Flickr": 150,
        "Squirrel": 150,
        "Reddit": 3,
    }

    # 激活函数的设置
    inner_act = {
        "Cora": lambda x: x,
        "Citeseer": torch.sigmoid,
        "ACM": lambda x: x,
        "Pubmed": lambda x: x,
        "BlogCatalog": lambda x: x,
        "Flickr": lambda x: x,
        "Squirrel": lambda x: x,
        "Reddit": lambda x: x,
    }

    # 更新图结构的周期
    udp = {
        "Cora": 10,
        "Citeseer": 40,
        "ACM": 40,
        "Pubmed": 10,
        "BlogCatalog": 40,
        "Flickr": 40,
        "Squirrel": 40,
        "Reddit": 40,
    }

    # 各个数据集的节点比例
    node_ratios = {
        "Cora": [1],
        "Citeseer": [0.3],
        "ACM": [0.3],
        "Pubmed": [0.5],
        "BlogCatalog": [1],
        "Flickr": [0.3],
        "Squirrel": [0.3],
        "Reddit": [0.01],
    }

    # 边的添加比例
    add_edge_ratio = {
        "Cora": 0.5,
        "Citeseer": 0.5,
        "ACM": 0.5,
        "Pubmed": 0.5,
        "BlogCatalog": 0.5,
        "Flickr": 0.5,
        "Reddit": 0.005,
    }

    # 边的删除比例
    del_edge_ratios = {
        "Cora": [0.01],
        "Citeseer": [0.005],
        "ACM": [0.005],
        "Pubmed": [0.005],
        "BlogCatalog": [0.005],
        "Flickr": [0.005],
        "Reddit": [0.02],
    }

    # 结构学习的轮数
    gsl_epochs_list = {
        "Cora": [5],
        "Citeseer": [5],
        "ACM": [10],
        "Pubmed": [3],
        "BlogCatalog": [10],
        "Flickr": [10],
        "Reddit": [1],
    }

    # 各数据集的正则化参数
    regularization = {
        "Cora": 1,
        "Citeseer": 0,
        "ACM": 0,
        "Pubmed": 1,
        "BlogCatalog": 0,
        "Flickr": 0,
        "Reddit": 0,
    }

    # 数据集来源
    source = {
        "Cora": "dgl",
        "Citeseer": "dgl",
        "ACM": "sdcn",
        "Pubmed": "dgl",
        "BlogCatalog": "cola",
        "Flickr": "cola",
        "Reddit": "dgl",
    }

    datasets = [args.dataset]  # 使用用户指定的数据集

    # 针对每个数据集，选择不同的模型版本
    for ds in datasets:
        if ds == "Reddit":
            hole = HoLe_batch  # 使用批处理模型
        else:
            hole = HoLe  # 使用标准模型

        # 遍历结构学习的不同轮数
        for gsl_epochs in gsl_epochs_list[ds]:
            runs = 1  # 运行次数

            # 遍历GNN层数配置
            for n_gnn_layer in n_gnn_layers[ds]:
                # 遍历预训练轮数
                for pre_epoch in pre_epochs[ds]:
                    # 遍历边删除比例
                    for del_edge_ratio in del_edge_ratios[ds]:
                        # 遍历节点比例
                        for node_ratio in node_ratios[ds]:
                            # 保存最终参数
                            final_params["dim"] = dim
                            final_params["n_gnn_layers"] = n_gnn_layer
                            final_params["n_lin_layers"] = n_lin_layers
                            final_params["lr"] = lr[ds]
                            final_params["pre_epochs"] = pre_epoch
                            final_params["epochs"] = epochs[ds]
                            final_params["udp"] = udp[ds]
                            final_params["inner_act"] = inner_act[ds]
                            final_params["add_edge_ratio"] = add_edge_ratio[ds]
                            final_params["node_ratio"] = node_ratio
                            final_params["del_edge_ratio"] = del_edge_ratio
                            final_params["gsl_epochs"] = gsl_epochs

                            # 获取当前时间，用于文件名
                            time_name = get_str_time()
                            save_file = f"results/hole/hole_{ds}_gnn_{n_gnn_layer}_gsl_{gsl_epochs}_{time_name[:9]}.csv"

                            # 加载数据集，包括图结构、标签和聚类数
                            graph, labels, n_clusters = load_data(
                                dataset_name=ds,
                                source=source[ds],
                                verbosity=2,
                            )
                            features = graph.ndata["feat"]  # 获取节点特征
                            if ds in ("Cora", "Pubmed"):
                                graph.ndata["feat"][(features - 0.0) > 0.0] = 1.0  # 特殊处理某些数据集的特征

                            # 获取图的邻接矩阵，格式为CSR稀疏矩阵
                            adj_csr = graph.adj_external(scipy_fmt="csr")
                            # 计算图的边数总和
                            adj_sum_raw = adj_csr.sum()
                            # 获取图的边
                            edges = graph.edges()
                            # 将节点特征转换为LIL格式的稀疏矩阵
                            features_lil = sp.lil_matrix(features)

                            # 将当前数据集的相关参数存储到final_params字典中
                            final_params["dataset"] = ds

                            # 模型热身文件名，用于判断是否已有预训练模型
                            warmup_filename = f"hole_{ds}_run_gnn_{n_gnn_layer}"

                            # 如果没有找到热身文件，进行模型的热身训练
                            if not check_modelfile_exists(warmup_filename):
                                print("warmup first")
                                # 初始化模型
                                model = hole(
                                    hidden_units=[dim],  # 隐藏层维度
                                    in_feats=features.shape[1],  # 输入特征维度
                                    n_clusters=n_clusters,  # 聚类数
                                    n_gnn_layers=n_gnn_layer,  # GNN层数
                                    n_lin_layers=n_lin_layers,  # 线性层数
                                    lr=lr[ds],  # 学习率
                                    n_pretrain_epochs=pre_epoch,  # 预训练轮数
                                    n_epochs=epochs[ds],  # 训练轮数
                                    norm="sym",  # 是否使用对称归一化
                                    renorm=True,  # 是否重新归一化
                                    tb_filename=f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                    # tensorboard文件名
                                    warmup_filename=warmup_filename,  # 预热模型文件名
                                    inner_act=inner_act[ds],  # 内部激活函数
                                    udp=udp[ds],  # 更新周期
                                    regularization=regularization[ds],  # 正则化参数
                                )

                                # 执行热身训练
                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio[ds],  # 边添加比例
                                    node_ratio=node_ratio,  # 节点比例
                                    del_edge_ratio=del_edge_ratio,  # 边删除比例
                                    gsl_epochs=0,  # 结构学习轮数
                                    labels=labels,  # 节点标签
                                    adj_sum_raw=adj_sum_raw,  # 邻接矩阵边的总和
                                    load=False,  # 是否加载已有模型
                                    dump=dump,  # 是否保存模型
                                )

                            # 随机种子列表，用于不同实验运行
                            seed_list = [
                                random.randint(0, 999999) for _ in range(runs)
                            ]
                            # 针对不同的运行次数进行训练
                            for run_id in range(runs):
                                final_params["run_id"] = run_id  # 记录运行ID
                                seed = seed_list[run_id]  # 设置随机种子
                                final_params["seed"] = seed  # 存储随机种子

                                # 针对Citeseer数据集需要重置模型，以避免过拟合
                                reset = ds == "Citeseer"

                                # 初始化模型
                                model = hole(
                                    hidden_units=[dim],
                                    in_feats=features.shape[1],
                                    n_clusters=n_clusters,
                                    n_gnn_layers=n_gnn_layer,
                                    n_lin_layers=n_lin_layers,
                                    lr=lr[ds],
                                    n_pretrain_epochs=pre_epoch,
                                    n_epochs=epochs[ds],
                                    norm="sym",
                                    renorm=True,
                                    tb_filename=f"{ds}_gnn_{n_gnn_layer}_node_{node_ratio}_{add_edge_ratio[ds]}_{del_edge_ratio}_gsl_{gsl_epochs}_pre_ep{pre_epoch}_ep{epochs[ds]}_dim{dim}_{random.randint(0, 999999)}",
                                    warmup_filename=warmup_filename,
                                    inner_act=inner_act[ds],
                                    udp=udp[ds],
                                    reset=reset,  # 是否重置模型
                                    regularization=regularization[ds],
                                    seed=seed,
                                )

                                # 训练模型
                                model.fit(
                                    graph=graph,
                                    device=device,
                                    add_edge_ratio=add_edge_ratio[ds],
                                    node_ratio=node_ratio,
                                    del_edge_ratio=del_edge_ratio,
                                    gsl_epochs=gsl_epochs,  # 结构学习轮数
                                    labels=labels,
                                    adj_sum_raw=adj_sum_raw,
                                    load=True,  # 是否加载模型
                                    dump=dump,  # 是否保存模型
                                )

                                # 获取训练后的嵌入表示
                                with torch.no_grad():
                                    z_detached = model.get_embedding()  # 获取节点嵌入
                                    Q = model.get_Q(z_detached)  # 获取聚类概率
                                    q = Q.detach().data.cpu().numpy().argmax(1)  # 获取聚类标签

                                # 使用多个指标进行评估
                                (
                                    ARI_score,  # 调整兰德指数
                                    NMI_score,  # 归一化互信息
                                    AMI_score,  # 调整互信息
                                    ACC_score,  # 准确率
                                    Micro_F1_score,  # 微平均F1得分
                                    Macro_F1_score,  # 宏平均F1得分
                                    purity,  # 纯度
                                ) = evaluation(labels, q)

                                # 打印评估结果
                                print("\n"
                                      f"ARI:{ARI_score}\n"
                                      f"NMI:{NMI_score}\n"
                                      f"AMI:{AMI_score}\n"
                                      f"ACC:{ACC_score}\n"
                                      f"Micro F1:{Micro_F1_score}\n"
                                      f"Macro F1:{Macro_F1_score}\n"
                                      f"purity_score:{purity}\n")

                                # 保存评估结果到final_params字典中
                                final_params["qARI"] = ARI_score
                                final_params["qNMI"] = NMI_score
                                final_params["qACC"] = ACC_score
                                final_params["qMicroF1"] = Micro_F1_score
                                final_params["qMacroF1"] = Macro_F1_score
                                final_params["qPurity"] = Macro_F1_score

                                # 如果有保存文件，写入CSV文件
                                if save_file is not None:
                                    csv2file(
                                        target_path=save_file,
                                        thead=list(final_params.keys()),  # CSV表头
                                        tbody=list(final_params.values()),  # CSV内容
                                    )
                                    print(f"write to {save_file}")  # 提示文件写入成功

                                # 打印最终参数
                                print(final_params)
