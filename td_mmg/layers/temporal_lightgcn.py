# coding=utf-8

import torch
import torch.nn as nn
import dgl
import dgl.function as fn


class TemporalLightGCNLayer(nn.Module):
    """
    时序衰减图卷积层
    考虑时间衰减因素的图卷积操作
    """
    
    def __init__(self):
        super(TemporalLightGCNLayer, self).__init__()
        # 初始化可学习的时间衰减率参数
        self.decay_lam = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
    
    def forward(self, g, h_dict):
        """
        前向传播
        :param g: DGL图对象，边特征包含 'dt' 和 'norm'
        :param h_dict: 同构图传入 tensor；异构图传入 dict（如 {'user': ..., 'item': ...}）
        :return: 同构图返回 tensor；异构图返回 dict
        """
        with g.local_scope():
            # 对可学习参数应用软阈值，保证其为正数
            lam = torch.relu(self.decay_lam) + 1e-4

            # 同构图：直接在 g.ndata 上使用 tensor
            if g.is_homogeneous:
                h = h_dict
                g.ndata['h'] = h

                if 'dt' not in g.edata:
                    g.edata['dt'] = torch.zeros(g.num_edges(), dtype=torch.float32, device=g.device)
                if 'norm' not in g.edata:
                    g.edata['norm'] = torch.ones(g.num_edges(), dtype=torch.float32, device=g.device)

                time_decay = torch.exp(-lam * g.edata['dt'])
                g.edata['temporal_weight'] = g.edata['norm'] * time_decay

                g.update_all(fn.u_mul_e('h', 'temporal_weight', 'm'), fn.sum('m', 'h_new'))
                return g.ndata['h_new']

            # 异构图：按边类型分别计算 temporal_weight
            g.ndata['h'] = h_dict
            for etype in g.etypes:
                if 'dt' not in g.edges[etype].data:
                    g.edges[etype].data['dt'] = torch.zeros(g.num_edges(etype), dtype=torch.float32, device=g.device)
                if 'norm' not in g.edges[etype].data:
                    g.edges[etype].data['norm'] = torch.ones(g.num_edges(etype), dtype=torch.float32, device=g.device)

                time_decay = torch.exp(-lam * g.edges[etype].data['dt'])
                g.edges[etype].data['temporal_weight'] = g.edges[etype].data['norm'] * time_decay

            message_func = fn.u_mul_e('h', 'temporal_weight', 'm')
            reduce_func = fn.sum('m', 'h_new')
            g.multi_update_all({etype: (message_func, reduce_func) for etype in g.etypes}, cross_reducer='sum')
            return g.ndata['h_new']
