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
    
    def forward(self, g, h):
        """
        前向传播
        :param g: DGL图对象，包含边特征 'dt' 和 'norm'
        :param h: 输入节点特征
        :return: 输出节点特征
        """
        with g.local_scope():
            # 将输入特征赋给图
            g.ndata['h'] = h
            
            # 对可学习参数应用软阈值，保证其为正数
            lam = torch.relu(self.decay_lam) + 1e-4
            
            # 计算时间衰减权重
            time_decay = torch.exp(-lam * g.edata['dt'])
            
            # 计算最终的消息传递权重
            temporal_weight = g.edata['norm'] * time_decay
            g.edata['temporal_weight'] = temporal_weight
            
            # 使用DGL的update_all机制进行消息传递
            g.update_all(
                fn.u_mul_e('h', 'temporal_weight', 'm'),
                fn.sum('m', 'h_new')
            )
            
            # 返回更新后的节点特征
            return g.ndata['h_new']
