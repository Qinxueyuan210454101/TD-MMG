# coding: utf-8

import os
import pandas as pd
import numpy as np
import torch
from logging import getLogger
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.configurator import Config
import platform

def convert_freedom_dataset_to_common(split_dataset, num_users, mask_datasets):
    split_df = split_dataset.df

    user_field = split_dataset.config['USER_ID_FIELD']
    item_field = split_dataset.config['ITEM_ID_FIELD']

    # group by user_field

    user_item_edges = np.array(split_df[[user_field, item_field]].values, dtype=np.int64)

    # convert to dict user=>items
    user_items_dict = split_df.groupby(user_field)[item_field].apply(list).to_dict()
    for user_index in range(num_users):
        if user_index not in user_items_dict:
            user_items_dict[user_index] = []


    mask_dfs = [mask_dataset.df for mask_dataset in mask_datasets]
    mask_df = pd.concat(mask_dfs)

    mask_user_items_dict = mask_df.groupby(user_field)[item_field].apply(list).to_dict()
    for user_index in range(num_users):
        if user_index not in mask_user_items_dict:
            mask_user_items_dict[user_index] = []

    return user_item_edges, user_items_dict, mask_user_items_dict


def load_data(dataset):
    config_dict = {}
    config = Config("FREEDOM", dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    # 重构数据切分逻辑：按时间戳切分
    train_dataset, valid_dataset, test_dataset, dt_norm = split_by_timestamp(dataset, config)
    
    logger.info('\n====Training====\n' + str(train_dataset))
    logger.info('\n====Validation====\n' + str(valid_dataset))
    logger.info('\n====Testing====\n' + str(test_dataset))

    num_users = dataset.user_num
    num_items = dataset.item_num

    train_user_item_edges, train_user_items_dict, train_mask_user_items_dict = convert_freedom_dataset_to_common(train_dataset, num_users, [valid_dataset, test_dataset])
    valid_user_item_edges, valid_user_items_dict, valid_mask_user_items_dict = convert_freedom_dataset_to_common(valid_dataset, num_users, [train_dataset, test_dataset])
    test_user_item_edges, test_user_items_dict, test_mask_user_items_dict = convert_freedom_dataset_to_common(test_dataset, num_users, [train_dataset, valid_dataset])


    v_feat, t_feat = None, None
    if not config['end2end'] and config['is_multimodal_model']:
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        # if file exist?
        v_feat_file_path = os.path.join(dataset_path, config['vision_feature_file'])
        t_feat_file_path = os.path.join(dataset_path, config['text_feature_file'])
        if os.path.isfile(v_feat_file_path):
            v_feat = torch.from_numpy(np.load(v_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)
        if os.path.isfile(t_feat_file_path):
            t_feat = torch.from_numpy(np.load(t_feat_file_path, allow_pickle=True)).type(torch.FloatTensor)

        assert v_feat is not None or t_feat is not None, 'Features all NONE'

    return train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat, dt_norm


def split_by_timestamp(dataset, config):
    """
    按时间戳切分数据：
    - 每个用户最后一次交互作为测试集
    - 倒数第二次交互作为验证集
    - 其余作为训练集
    并计算归一化时间差
    """
    df = dataset.df.copy()
    user_field = config['USER_ID_FIELD']
    item_field = config['ITEM_ID_FIELD']
    time_field = config['TIME_FIELD']
    
    # 按用户ID和时间戳排序
    df = df.sort_values([user_field, time_field])
    
    # 为每个用户分配数据类型：0=训练, 1=验证, 2=测试
    df['split_label'] = 0
    
    # 按用户分组，获取每个用户的交互记录
    grouped = df.groupby(user_field)
    
    for user_id, group in grouped:
        if len(group) >= 2:
            # 最后一条作为测试集
            df.loc[group.index[-1], 'split_label'] = 2
            # 倒数第二条作为验证集
            df.loc[group.index[-2], 'split_label'] = 1
        elif len(group) == 1:
            # 只有一条记录，作为测试集
            df.loc[group.index[0], 'split_label'] = 2
    
    # 分割数据集
    train_df = df[df['split_label'] == 0].copy()
    valid_df = df[df['split_label'] == 1].copy()
    test_df = df[df['split_label'] == 2].copy()
    
    # 移除分割标签
    train_df = train_df.drop('split_label', axis=1)
    valid_df = valid_df.drop('split_label', axis=1)
    test_df = test_df.drop('split_label', axis=1)
    
    # 计算归一化时间差
    dt_norm = compute_time_delta(train_df, time_field)
    
    # 包装为RecDataset
    train_dataset = dataset.copy(train_df)
    valid_dataset = dataset.copy(valid_df)
    test_dataset = dataset.copy(test_df)
    
    return train_dataset, valid_dataset, test_dataset, dt_norm

def compute_time_delta(train_df, time_field):
    """
    计算训练集的归一化时间差
    delta_t = T_max - timestamp
    归一化到 [0, 10] 范围
    """
    if time_field not in train_df.columns:
        # 如果没有时间字段，返回None
        return None
    
    # 计算T_max
    T_max = train_df[time_field].max()
    
    # 计算时间差
    delta_t = T_max - train_df[time_field]
    
    # 归一化到 [0, 10]
    if delta_t.max() > delta_t.min():
        dt_norm = 10 * (delta_t - delta_t.min()) / (delta_t.max() - delta_t.min())
    else:
        dt_norm = delta_t * 0  # 所有值都相同，设为0
    
    # 转换为numpy数组
    dt_norm = dt_norm.values
    
    return dt_norm


import dgl

def dgl_add_all_reversed_edges(g):
    edge_dict = {}
    for etype in list(g.canonical_etypes):
        col, row = g.edges(etype=etype)
        edge_dict[etype] = (col, row)

        if etype[0] != etype[2]:
            new_etype = (etype[2], "r.{}".format(etype[1]), etype[0])
            edge_dict[new_etype] = (row, col)

    new_g = dgl.heterograph(edge_dict)

    for key in g.ndata:
        print("key = ", key)
        new_g.ndata[key] = g.ndata[key]

    return new_g


def build_hetero_graph(user_item_edges, num_users, num_items, dt_norm=None):
    edge_dict = {}

    user_item_edges = (user_item_edges[:, 0], user_item_edges[:, 1])
    edge_dict[("user", "user_item", "item")] = user_item_edges

    if True:

        item_image_edges = (torch.arange(num_items), torch.arange(num_items))
        edge_dict[("item", "item_image", "item_image")] = item_image_edges

        item_text_edges = (torch.arange(num_items), torch.arange(num_items))
        edge_dict[("item", "item_text", "item_text")] = item_text_edges

        g = dgl.heterograph(edge_dict, num_nodes_dict={"user": num_users, "item": num_items, "item_image": num_items, "item_text": num_items})
    else:
        g = dgl.heterograph(edge_dict, num_nodes_dict={"user": num_users, "item": num_items})

    g = dgl_add_all_reversed_edges(g)

    # 计算度数归一化系数
    # 计算用户度数 (出度，返回大小为 num_users)
    user_deg = g.out_degrees(etype='user_item').float()
    # 计算物品度数 (入度，返回大小为 num_items)
    item_deg = g.in_degrees(etype='user_item').float()
    
    # 防止除以 0，截断最小值为 1.0
    user_deg = torch.clamp(user_deg, min=1.0)
    item_deg = torch.clamp(item_deg, min=1.0)
    
    # 计算 1 / sqrt(d_u * d_i)
    src, dst = g.edges(etype='user_item')
    norm = 1.0 / torch.sqrt(user_deg[src] * item_deg[dst])
    norm = norm.unsqueeze(1)  # 增加维度以便后续广播
    
    # 为User-Item边添加度数归一化系数
    g.edges["user_item"].data["norm"] = norm
    # 为反向边添加相同的度数归一化系数
    g.edges["r.user_item"].data["norm"] = norm
    
    # 添加时间差特征
    if dt_norm is not None:
        dt_tensor = torch.tensor(dt_norm, dtype=torch.float32).unsqueeze(1)
    else:
        # 验证集和测试集的时间差默认为0
        dt_tensor = torch.zeros((g.num_edges(etype="user_item"), 1), dtype=torch.float32)
    
    # 为User-Item边添加时间差特征
    g.edges["user_item"].data["dt"] = dt_tensor
    # 为反向边添加相同的时间差特征
    g.edges["r.user_item"].data["dt"] = dt_tensor

    return g


def load_hetero_data(dataset):
    train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat, dt_norm = load_data(dataset)

    train_hetero_g = build_hetero_graph(train_user_item_edges, num_users, num_items, dt_norm)
    valid_hetero_g = build_hetero_graph(valid_user_item_edges, num_users, num_items)
    test_hetero_g = build_hetero_graph(test_user_item_edges, num_users, num_items)

    return train_user_item_edges, valid_user_item_edges, test_user_item_edges, train_user_items_dict, train_mask_user_items_dict, valid_user_items_dict, valid_mask_user_items_dict, test_user_items_dict, test_mask_user_items_dict, num_users, num_items, v_feat, t_feat, train_hetero_g, valid_hetero_g, test_hetero_g, dt_norm