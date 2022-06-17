import torch
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

def event_mention_mask(input_ids, event1, event1_mask, event2, event2_mask):
    """Mask模块, 对事件描述进行遮蔽处理"""

    new_input_ids = input_ids.clone()
    for i in range(input_ids.size(0)):
        event1_span = torch.masked_select(event1[i], event1_mask[i].bool())
        event2_span = torch.masked_select(event2[i], event2_mask[i].bool())
        new_input_ids[i, event1_span] = 103
        new_input_ids[i, event2_span] = 103
    return new_input_ids

def mention_one_mask(ids, mask, e1, e1_mask, e2, e2_mask):
    batch_size  = ids.size(0)
    new_ids, new_mask = ids.clone(), mask.clone()
    new_e1, new_e1_mask = e1.clone(), e1_mask.clone()
    new_e2, new_e2_mask = e2.clone(), e2_mask.clone()
    new_e1[:, 1:], new_e1_mask[:, 1:] = 0, 0
    new_e2[:, 1:], new_e2_mask[:, 1:] = 0, 0

    for i in range(batch_size):
        event1_span = torch.masked_select(e1[i], e1_mask[i].bool())
        event2_span = torch.masked_select(e2[i], e2_mask[i].bool())
        mask_span = [event1_span[0], event2_span[0]]
        new_ids[i, mask_span] = 103
        pad_span = torch.cat([event1_span[1:], event2_span[1:]])
        new_ids[i, pad_span] = 0
        new_mask[i, pad_span] = 0
    
    return new_ids, new_mask, new_e1, new_e1_mask, new_e2, new_e2_mask
 
def get_mask(path_len):
    """
    根据关系路径长度生成注意力掩码
    :param path_len: 关系路径长度列表, 1维list
    :return: 注意力掩码(batch_size, node_num, node_num)
    """

    batch_size = len(path_len)
    node_num = max(path_len)
    attn_mask = torch.ones((batch_size, node_num, node_num)).bool()
    for i, l in enumerate(path_len):
        attn_mask[i, :, :l] = False
    return attn_mask

def get_adj(path_len):
    """根据关系路径长度生成邻接矩阵

    Args:
        path_len (1维list): 关系路径长度列表, 1维list

    Returns:
        _type_: 邻接矩阵(batch_size, node_num, node_num)
    """    

    # # 获取严格意义的邻接矩阵
    # node_num = max(path_len)
    # ones = torch.ones(len(path_len), node_num, node_num)
    # adj = torch.triu(ones, -1) * torch.tril(ones, 1)
    # for i, l in enumerate(path_len):
    #     adj[i, l:, :] = 0
    #     adj[i, :, l:] = 0
    # return adj

    # # 获取对角矩阵
    node_num = max(path_len)
    adj_list = []
    for l in path_len:
        adj = torch.eye(node_num)
        adj[l:, l:] = 0
        adj_list.append(adj)
    return torch.stack(adj_list)

def get_key(ids, e1, e2):
    """根据未扩充的数据生成result_dict中的key值"""
    
    str_list = []
    for x in [ids, e1, e2]:
        x = x if type(x) is list else x.tolist()
        str_list.append('_'.join([str(num) for num in x]))
    key = ' '.join(str_list)
    return key
