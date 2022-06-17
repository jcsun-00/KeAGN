import os
import random
import datetime
import pytz

import numpy as np
import torch

from models.model import *
from models.LSIN import BertCausalModel

def show_time():
    """ 打印当前时间 """

    print('time: ' + \
        datetime.datetime.now(pytz.timezone('PRC')).strftime("%Y-%m-%d %H:%M:%S"))


def show_pid():
    """ 打印当前程序pid """

    pid = os.getpid()
    print('pid:', pid)


def setup_seed(seed):
    """设置随机数种子, 确保结果可重复"""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def topic_split(dataset, test_topic):
    """划分训练集 (测试集以外的部分) 和测试集 """

    train_set, test_set = [], []
    for data in dataset:
        t = data[0]
        if t.split('/')[-2] in test_topic:
            test_set.append(data)
        else:
            train_set.append(data)

    return train_set, test_set


def train_test_split(dataset, index):
    """划分训练集 (测试集以外的部分) 和测试集 """

    train_set, test_set = [], []
    for i, data in enumerate(dataset):
        if i in index:
            train_set.append(data)
        else:
            test_set.append(data)
    return train_set, test_set


def data_augment(train_set):

    rel_dict = { 'NULL': 'NULL','null': 'null', 'FALLING_ACTION': 'PRECONDITION', 'PRECONDITION': 'FALLING_ACTION'}
    new_data = []
    for item in train_set:
        doc_name, sen_ids, sen_ids, ids_span1, ids_span2, \
            rel, graph1, graph2, objects1_ids, objects2_ids, \
            sdp_path_ids_span, rel_path_ids = item
        new_rel = rel_dict[rel]
        new_rel_path_ids = rel_path_ids[-1:] + rel_path_ids[1:-1] + rel_path_ids[0:1]
        new_item = [
            doc_name, sen_ids, sen_ids, ids_span2, ids_span1,  \
            new_rel, graph2, graph1, objects2_ids, objects1_ids,  \
            sdp_path_ids_span, new_rel_path_ids
        ]
        new_data.append(item)
        new_data.append(new_item)
    return new_data


def neg_sampling(train_set, ratio):
    """对训练集的负样本进行降采样"""
    
    filter = []
    for d in train_set:
        if d[5]=='NULL' and random.random()>ratio: 
            continue
        filter.append(d)
    return filter


def compute_f1(gold, predicted):
    """
    :param gold: 真实值
    :param predicted: 预测值
    :return: Precision, Recall, F1 score 
    """

    c_predict = np.sum(predicted!=0) # 预测为因果关系的数量
    c_gold = np.sum(gold!=0) # 真实标签为因果关系的数量
    c_correct = np.sum((predicted==gold) & (gold==1)) # 预测为因果关系且预测正确的数量

    p = c_correct / (c_predict + 1e-100)
    r = c_correct / c_gold
    f = 2 * p * r / (p + r + 1e-100)
    a = np.sum(predicted==gold) / len(gold)

    print(f'Correct: {c_correct}, Pedicted: {c_predict}, Golden: {c_gold}, Accuracy:{a}')

    return {'precision': p, 'recall': r, 'f1_score': f}


def get_model(name, device, path=None):
    """ 根据模型名称`name`加载或创建对应模型 """

    try:
        model = torch.load(path)
        print(f'Sucessfully loaded the model from {path}')
    except:
        eval_line = ""
        if 'BertCausalModel' in name:
            eval_line = name[:-1]+', device=device)'
        elif name[-1] != ')':
            eval_line = name+'()'
        else:
            eval_line = name
        model = eval(eval_line)
    finally:
        return model.to(device)


def get_optimizer(model, learning_rate):
    """获取AdamW优化器, 并针对所有层的bias以及层归一化的weight取消权重衰减)"""

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)


def model_forward(model, batch):
    """ 根据model类型从batch中选择对应数据进行正向传播 """

    sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, \
        data_y, graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
        sdp_path, sdp_path_concept, sdp_path_concept_len, mask_sdp_path_concept = batch

    opt = None
    
    if type(model).__name__=='BertCausalModel':
        opt = model(sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, \
                    graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
                    sdp_path, sdp_path_concept, mask_sdp_path_concept)
    elif type(model).__name__ in ['BaseBertModel', 'MasGModel']:
        opt = model(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
    elif type(model).__name__ in ['BertDKModel']:
        opt = model(sentences_s, mask_s, event1, event1_mask, event2, event2_mask, \
                    graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2)
    elif type(model).__name__ in ['BertRKModel']:
        opt = model(sentences_s, mask_s, event1, event1_mask, event2, event2_mask, \
                    sdp_path_concept, sdp_path_concept_len, mask_sdp_path_concept)
    elif type(model).__name__ in ['LSINModel']:
        opt = model(sentences_s, mask_s, event1, event1_mask, event2, event2_mask, \
                    graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
                    sdp_path_concept, sdp_path_concept_len, mask_sdp_path_concept)
    elif type(model).__name__ in ['MixedModel']:
        opt = model(sentences_s, mask_s, event1, event1_mask, event2, event2_mask, \
                    graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
                    sdp_path_concept, sdp_path_concept_len, mask_sdp_path_concept)
    else:
        raise ValueError(f'No forwarding way about {type(model).__name__}')
    return opt
