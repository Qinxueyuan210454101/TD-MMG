import torch
import torch.nn as nn
from td_mmg.layers.common import MyMLP
from td_mmg.layers.mgdcf import MGDCF
from td_mmg.layers.temporal_lightgcn import TemporalLightGCNLayer




class MMMLP(nn.Module):
    def __init__(self, 
                 feat_drop_rate, 
                 item_v_in_channels=None,
                 item_v_hidden_channels_list=None,
                 item_t_in_channels=None,
                 item_t_hidden_channels_list=None,
                 item_hidden_channels_list=None,
                 *args, **kwargs):
        
        super().__init__()

        self.user_input_dropout = nn.Dropout(p=feat_drop_rate)
        self.item_input_dropout = nn.Dropout(p=feat_drop_rate)
        

        self.item_v_mlp = MyMLP(item_v_in_channels, item_v_hidden_channels_list,
                                activation="prelu", drop_rate=feat_drop_rate, bn=True, 
                                output_activation="prelu", output_drop_rate=feat_drop_rate, output_bn=True)
        self.item_v_input_dropout = nn.Dropout(p=feat_drop_rate)

        self.item_t_mlp = MyMLP(item_t_in_channels, item_t_hidden_channels_list,
                                activation="prelu", drop_rate=feat_drop_rate, bn=True, 
                                output_activation="prelu", output_drop_rate=feat_drop_rate, output_bn=True)
        self.item_t_input_dropout = nn.Dropout(p=feat_drop_rate)

        item_v_out_channels = item_v_hidden_channels_list[-1] if len(item_v_hidden_channels_list) > 0 else item_v_in_channels
        item_t_out_channels = item_t_hidden_channels_list[-1] if len(item_t_hidden_channels_list) > 0 else item_t_in_channels

        self.item_mlp = MyMLP(item_v_out_channels + item_t_out_channels, 
                              item_hidden_channels_list,
                              activation="prelu", drop_rate=feat_drop_rate, bn=True, 
                              output_activation="prelu", output_drop_rate=0.0, output_bn=True)


    def forward(self, item_v_feat, item_t_feat):

        item_v_h = self.item_v_input_dropout(item_v_feat)
        item_v_h = self.item_v_mlp(item_v_h)

        item_t_h = self.item_t_input_dropout(item_t_feat)
        item_t_h = self.item_t_mlp(item_t_h)

        item_h = torch.cat([item_v_h, item_t_h], dim=-1)
        item_h = self.item_mlp(item_h)      

        return item_h      

class MMMGDCF(nn.Module):
    def __init__(self, 
                #  k, 
                 k_e,
                 k_t,
                 k_v,
                 alpha, beta, 
                 input_feat_drop_rate,
                 feat_drop_rate,
                 user_x_drop_rate,
                 item_x_drop_rate, 
                 edge_drop_rate, z_drop_rate, 
                 user_in_channels=None,
                 user_hidden_channels_list=None,
                 item_v_in_channels=None,
                 item_v_hidden_channels_list=None,
                 item_t_in_channels=None,
                 item_t_hidden_channels_list=None,
                #  item_hidden_channels_list=None,
                 bn=True,
                 use_dual=False,
                 *args, **kwargs):
        
        super().__init__()

        # self.emb_mgdcf = MGDCF(4, alpha, beta, 0.0, edge_drop_rate, z_drop_rate)
        # self.t_mgdcf = MGDCF(k, alpha, beta, 0.0, edge_drop_rate, z_drop_rate)
        # self.v_mgdcf = MGDCF(3, alpha, beta, 0.0, edge_drop_rate, z_drop_rate)

 

        # 替换为时序衰减图卷积层
        # 先只用一层进行实验验证
        self.emb_mgdcf = TemporalLightGCNLayer() if k_e >= 0 else None
        # 暂时注释掉t_mgdcf和v_mgdcf
        # self.t_mgdcf = MGDCF(k_t, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_t >= 0 else None
        # self.v_mgdcf = MGDCF(k_v, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_v >= 0 else None
        self.t_mgdcf = None
        self.v_mgdcf = None

        # self.homo_t_mgdcf = MGDCF(1, alpha, beta, 0.0, edge_drop_rate, 0.0) if k_t >= 0 else None

        self.z_dropout = nn.Dropout(z_drop_rate)


        if user_hidden_channels_list is not None:
            raise Exception()

        self.user_gnn_input_dropout = nn.Dropout(user_x_drop_rate)
        self.item_gnn_input_dropout = nn.Dropout(item_x_drop_rate)


        self.use_dual = use_dual


        # self.t_mlp = nn.Sequential(
        #     nn.Dropout(input_feat_drop_rate),
        #     MyMLP(
        #         item_t_in_channels, 
        #         # item_t_hidden_channels_list,
        #         [item_t_hidden_channels_list[-1]],
        #         activation="prelu",
        #         drop_rate=feat_drop_rate,
        #         bn=True,
        #         output_activation="none",#"prelu",
        #         output_drop_rate=0.0,
        #         output_bn=False#True
        #     )
        # )

        # self.t_mlp = nn.Sequential(
        #     nn.Dropout(input_feat_drop_rate),
        #     MyMLP(
        #         item_t_in_channels, 
        #         item_t_hidden_channels_list,
        #         activation="prelu",
        #         drop_rate=feat_drop_rate,
        #         bn=True,
        #         output_activation="prelu",
        #         output_drop_rate=0.0,
        #         output_bn=True
        #     )
        # )

        self.t_mlp = nn.Sequential(
            nn.Dropout(input_feat_drop_rate),
            MyMLP(
                item_t_in_channels, 
                item_t_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=bn
            )
        )

        self.v_mlp = nn.Sequential(
            nn.Dropout(input_feat_drop_rate),
            MyMLP(
                item_v_in_channels, 
                item_v_hidden_channels_list,
                activation="prelu",
                drop_rate=feat_drop_rate,
                bn=True,
                output_activation="prelu",
                output_drop_rate=0.0,
                output_bn=bn
            )
        )



    def forward(self, g, user_embeddings, item_v_feat, item_t_feat, item_embeddings=None, item_item_g=None, return_all=False):


        encoded_t = self.t_mlp(item_t_feat)
        encoded_v = self.v_mlp(item_v_feat)

        num_users = user_embeddings.size(0)
        user_h = user_embeddings

        if self.emb_mgdcf is not None:
            # 构建节点特征字典
            h_dict = {
                'user': self.user_gnn_input_dropout(user_h),
                'item': torch.zeros_like(encoded_t) if item_embeddings is None else self.item_gnn_input_dropout(item_embeddings)
            }
            # 调用新的时序衰减图卷积层
            emb_h_dict = self.emb_mgdcf(g, h_dict)
            # 将字典转换为拼接的张量
            emb_h = torch.cat([emb_h_dict['user'], emb_h_dict['item']], dim=0)
        else:
            emb_h = None

        # 暂时置为None
        t_h = None
        v_h = None

        # 跳过item_item_g相关逻辑
        if item_item_g is not None:
            pass

      
        combined_h = None
        for i, h in enumerate([emb_h, t_h, v_h]):
            if h is not None:
                if combined_h is None:
                    combined_h = h
                else:
                    combined_h  = combined_h + h

        combined_h = self.z_dropout(combined_h)

        emb_h = self.z_dropout(emb_h) if emb_h is not None else None
        t_h = self.z_dropout(t_h) if t_h is not None else None
        v_h = self.z_dropout(v_h) if v_h is not None else None

        if return_all:
            return combined_h, emb_h, t_h, v_h, encoded_t, encoded_v
        else:
            return combined_h



            

        
