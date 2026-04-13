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
        :param g: DGL异构图对象，包含边特征 'dt' 和 'norm'
        :param h_dict: 输入节点特征字典，格式为 {'user': user_tensor, 'item': item_tensor}
        :return: 输出节点特征字典
        """
        with g.local_scope():
            # 将输入特征赋给图
            g.ndata['h'] = h_dict
            
            # 对可学习参数应用软阈值，保证其为正数
            lam = torch.relu(self.decay_lam) + 1e-4
            
            # 为每种边类型计算时间衰减权重
            for etype in ['user_item', 'r.user_item']:
                if etype in g.etypes:
                    # 计算时间衰减
                    time_decay = torch.exp(-lam * g.edges[etype].data['dt'])
                    # 计算最终的消息传递权重
                    temporal_weight = g.edges[etype].data['norm'] * time_decay
                    g.edges[etype].data['temporal_weight'] = temporal_weight
            
            # 定义消息传递函数
            message_func = fn.u_mul_e('h', 'temporal_weight', 'm')
            reduce_func = fn.sum('m', 'h_new')
            
            # 使用DGL的multi_update_all机制进行消息传递
            g.multi_update_all(
                {
                    'user_item': (message_func, reduce_func),
                    'r.user_item': (message_func, reduce_func)
                },
                cross_reducer='sum'
            )
            
            # 返回更新后的节点特征
            return g.ndata['h_new']
