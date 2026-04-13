# coding=utf-8

import torch
import torch.nn as nn
import dgl
import numpy as np
import dgl.function as fn
from td_mmg.layers.temporal_lightgcn import TemporalLightGCNLayer






class MGDCF(nn.Module):

    CACHE_KEY = "mgdcf_weight"

    def __init__(self, k, alpha, beta, x_drop_rate, edge_drop_rate, z_drop_rate, *args, **kwargs):
        """
        :param k: number of iterations.
        :param alpha: a hyperparameter of MGDCF.
        :param beta: a hyperparameter of MGDCF.
        :param x_drop_rate: dropout rate of input embeddings.
        :param edge_drop_rate: dropout rate of edge weights.
        :param z_drop_rate: dropout rate of output embeddings.
        """

        super().__init__(*args, **kwargs)

        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.gamma = torch.tensor(MGDCF.compute_gamma(alpha, beta, k)).float()

        self.gammas =[torch.tensor(MGDCF.compute_gamma(alpha, beta, i)).float() for i in range(1, k+1)]

        self.x_drop_rate = x_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.z_drop_rate = z_drop_rate

        self.x_dropout = nn.Dropout(x_drop_rate)
        self.edge_dropout = nn.Dropout(edge_drop_rate)
        self.z_dropout = nn.Dropout(z_drop_rate)
        
        # 初始化时序衰减图卷积层
        self.temporal_gcn = TemporalLightGCNLayer()


    @classmethod
    def compute_gamma(cls, alpha, beta, k):
        return np.power(beta, k) + alpha * np.sum([np.power(beta, i) for i in range(k)])

    # @classmethod
    # def build_homo_graph(cls, user_item_edges, num_users=None, num_items=None):

    #     user_index, item_index = user_item_edges.T

    #     if num_users is None:
    #         num_users = np.max(user_index) + 1

    #     if num_items is None:
    #         num_items = np.max(item_index) + 1

    #     num_homo_nodes = num_users + num_items
    #     homo_item_index = item_index + num_users
    #     src = user_index
    #     dst = homo_item_index

    #     g = dgl.graph((src, dst), num_nodes=num_homo_nodes)
    #     g =  dgl.add_reverse_edges(g)
    #     # Different from LightGCN, MGDCF considers self-loop
    #     g = dgl.add_self_loop(g)
    #     g = dgl.to_simple(g)
        
    #     return g

    @classmethod
    def build_sorted_homo_graph(cls, user_item_edges, num_users=None, num_items=None, dt_norm=None):

        user_index, item_index = user_item_edges.T

        if num_users is None:
            num_users = np.max(user_index) + 1

        if num_items is None:
            num_items = np.max(item_index) + 1

        user_index = torch.tensor(user_index)
        item_index = torch.tensor(item_index)

        num_homo_nodes = num_users + num_items
        homo_item_index = item_index + num_users

        # 创建用户-物品边
        src = user_index
        dst = homo_item_index
        num_user_item_edges = len(src)

        # 添加反向边（物品-用户）
        src = torch.concat([src, dst], dim=0)
        dst = torch.concat([dst, src], dim=0)

        # 添加自环边
        src = torch.concat([src, torch.arange(num_homo_nodes)], dim=0)
        dst = torch.concat([dst, torch.arange(num_homo_nodes)], dim=0)

        g = dgl.graph((src, dst), num_nodes=num_homo_nodes)

        # 添加时间差特征
        if dt_norm is not None:
            # 为用户-物品边添加dt
            user_item_dt = torch.tensor(dt_norm, dtype=torch.float32)
            # 反向边使用相同的dt
            reverse_dt = user_item_dt
            # 自环边的dt设为0
            self_loop_dt = torch.zeros(num_homo_nodes, dtype=torch.float32)
            # 合并所有边的dt
            all_dt = torch.concat([user_item_dt, reverse_dt, self_loop_dt], dim=0)
            g.edata['dt'] = all_dt

        assert g.num_edges() == src.size(0)
        
        return g


    @classmethod
    @torch.no_grad()
    def norm_adj(cls, g):

        CACHE_KEY = MGDCF.CACHE_KEY

        # if CACHE_KEY in g.edata:
        #     return

        
        degs = g.in_degrees()
        src_norm = degs.pow(-0.5)
        dst_norm = src_norm

        with g.local_scope():
            g.ndata["src_norm"] = src_norm
            g.ndata["dst_norm"] = dst_norm
            g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", CACHE_KEY))
            gcn_weight = g.edata[CACHE_KEY]

        g.edata[CACHE_KEY] = gcn_weight


    def forward(self, g, x, return_all=False):

        CACHE_KEY = MGDCF.CACHE_KEY
        MGDCF.norm_adj(g)

        # 确保图中有'dt'特征
        if 'dt' not in g.edata:
            # 如果没有'dt'特征，添加默认值0
            g.edata['dt'] = torch.zeros(g.num_edges(), dtype=torch.float32, device=g.device)
        
        # 确保图中有'norm'特征
        if 'norm' not in g.edata:
            # 如果没有'norm'特征，使用MGDCF计算的权重作为norm
            g.edata['norm'] = g.edata[CACHE_KEY]

        h0 = self.x_dropout(x)
        h = h0

        if return_all:
            h_list = []

        for _ in range(self.k):
            # 使用时序衰减图卷积层进行消息传递
            h = self.temporal_gcn(g, h)
            h = h * self.beta + h0 * self.alpha

            if return_all:
                h_list.append(h)

        if not return_all:
            h = h / self.gamma
            h = self.z_dropout(h)
            return h
        else:                
            h_list = [h / gamma for h, gamma in zip(h_list, self.gammas)]
            h_list = [self.z_dropout(h) for h in h_list]
            return h_list

        # if return_all:
        #     return h_list[-1], h_list
        # else:
        #     return h_list[-1]
       
