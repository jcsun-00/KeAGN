#===================================================
# 数据预处理:
# 1. 构建描述型知识图谱
# 2. 构建关系路径(rel_path)
#===================================================

from itertools import combinations, permutations # 用于列举两个事件不同的事件对
import collections
import random
import pickle

from tqdm import tqdm
from transformers import BertTokenizer
import numpy as np
import networkx as nx
import spacy

import util.event_to_concept as e2c # 自定义文件

RELATIONS = {
    'capableof': 'capable of',
    'isa': 'is a',
    'hasproperty': 'has property',
    'causes': 'causes', 
    'causesdesire': 'causes desire',
    'usedfor': 'used for',
    'hassubevent': 'has sub event',
    'hasprerequisite': 'has pre requisite', 
    'partof': 'part of',
    'hasa': 'has a',
    'receivesaction': 'receives action',
    'createdby': 'created by',
    'madeof': 'made of',
    'desires': 'desires'
}


def read_cpnet(comet_path):
    """读取ConceptNet文件与COMET生成文件, 返回边的集合"""

    subj2rel_obj = collections.defaultdict(list) # key为subject, value为relation+' '+object列表
    subj2rel = collections.defaultdict(set) # key为subject, value为relation集合
    edges = [] # 每个元素为 (subject, object)二元组
    edge_dict = collections.defaultdict(set) # 键为(subject, object), 值为rel

    with open('./data/ConcepNet/conceptnet_en.csv', 'r') as f:
        for line in f.readlines():
            rel, head, tail, _ = line.strip('\n\r').split('\t')
            edges.append((head, tail))
            edge_dict[(head, tail)].add(rel)
            subj2rel_obj[head].append((rel, tail))
            subj2rel[head].add(rel)

    with open(comet_path, 'r') as f:
        for line in f.readlines():
            subj, rel, obj = line.strip('\n\r').split('\t')
            assert rel in RELATIONS
            edges.append((subj, obj))
            edge_dict[(subj, obj)].add(rel)
            if subj not in subj2rel or rel not in subj2rel[subj]:
                subj2rel_obj[subj].append((rel, obj))
                subj2rel[subj].add(rel)

    return subj2rel_obj, edges, edge_dict


def build_graph(subj2rel_obj, zero_concept, cpnet_graph, concept_num=1):
    """给定`0-hop概念`, 生成对应的知识图谱"""

    adj_matrix = np.identity(15) # 知识图谱的邻接矩阵 15=1(0-hop概念)+14(1-hop概念)
    relations, objects = [], [] # 14种关系及其每个关系对应的概念
    rel_objs = collections.defaultdict(list) # key为关系, value为object列表

    for r, obj in subj2rel_obj[zero_concept]: 
        rel_objs[r].append(obj)

    for i, rel in enumerate(RELATIONS):
        objs = rel_objs[rel] # 获取该关系对应的概念列表
        if len(objs)<concept_num: objs += ['[PAD]']*(concept_num-len(objs))
        obj = random.sample(objs, concept_num) # 随机选取一个概念
        relations.append(rel)
        objects.append(obj) if type(obj) is not list else objects.extend(obj)
        adj_matrix[0][i+1] = 1
        adj_matrix[i+1][0] = 1

    for i, j in combinations(range(1, 15), r=2):
        if adj_matrix[i][j]==1: continue
        try:
            path_len = nx.shortest_path_length(cpnet_graph, objects[i-1], objects[j-1])
            status = int(path_len>=2)
        except nx.NetworkXNoPath:
            status = 0
        finally:
            adj_matrix[i][j] = status
            adj_matrix[j][i] = status
    return relations, objects, adj_matrix


def build_seq(subj2rel_obj, zero_concept):
    """给定`0-hop概念`, 生成对应的知识序列"""

    rel_objs = collections.defaultdict(list) # key为关系, value为object列表

    for r, obj in subj2rel_obj[zero_concept]: 
        rel_objs[r].append(obj)

    return rel_objs


def get_key(ids, e1_span, e2_span):
    """根据id序列及e1,e2在序列中的位置生成key值"""
    
    str_list = []
    for x in [ids, e1_span, e2_span]:
        str_list.append('_'.join([str(num) for num in x]))
    key = ' '.join(str_list)
    return key


def filter(ids_dict, data):
    """对生成数据进行进一步筛选"""

    new_data = []
    for item, _ in zip(data, tqdm(data)):
        sen_ids, ids_span1, ids_span2, rel = item[2:6]
        key = get_key(sen_ids, ids_span1, ids_span2)
        if rel in ids_dict[key]:
            new_data.append(item)
            if rel=='NULL':
                ids_dict[key].remove('NULL')
    return new_data


def tokenize(tokenizer:BertTokenizer, words:list, join=False):
    """使用tokenizer进行分词

    Args:
        words (list): 单词列表
        join (bool, optional): 分词前是否对列表进行拼接. Defaults to False.

    Returns:
        list: 句子的分词结果(1维)或单词列表的分词结果(2维)
    """

    if join:
        seq = ' '.join(words).replace('_', ' ').lower()
        ids = tokenizer(seq)['input_ids']
        return ids
    else:
        words = [word.replace('_',' ') for word in words]
        ids_list = [tokenizer(word)['input_ids'][1:-1] for word in words]
        return ids_list


if __name__ == '__main__':

    # 设定随机数种子
    seed = 0
    np.random.seed(seed)
    random.seed(seed)

    # 是否保存预处理后的数据
    save_mode = True
    # 加载数据
    ds = 'CTB'
    doc_data_path = f'./data/document/document_{ds}.pickle'
    comet_path = f'data/ConcepNet/comet_{ds}.txt'
    save_path = f'./data/data_v4.2_{ds}_{seed}.pickle'

    # 读取document_*.pickle
    with open(doc_data_path, 'rb') as f:
        documents = pickle.load(f)

    # 加载模型
    nlp = spacy.load(name="en_core_web_sm")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # 读取ConceptNet和COMET数据, 形成外部知识图谱
    MATCH_DICT = e2c.get_match_dict(ds)
    subj2rel_obj, cpnet_edges, edge_dict = read_cpnet(comet_path)
    cpnet_graph = nx.Graph(cpnet_edges)

    # 声明全局数据
    data_set = []
    count_event_pair = {'total':0, 'causal':0} # 因果事件对的数量
    ids_dict = collections.defaultdict(set) # 统计句子token_id序列, e1_span和e2_span不同的数量
    count_drop_causal = {'跨句':0}
    event_dk_dict = {} # 统计事件及描述性知识的对应关系

    for doc_name, _ in zip(documents, tqdm(documents)):
        [all_token, ecb_star_events, _, eval_data] = documents[doc_name]

        ep_rel = {} # 记录具有因果关系的事件对及其关系, key为事件对, value为关系
        for eval_item in eval_data:
            ep_rel[(eval_item[0], eval_item[1])] = eval_item[2]
        ep_sen_rel = collections.defaultdict(list) # 记录已经遍历过的事件对

        # 遍历文档内的事件对 (忽略由相同事件组成的事件对) 循环次数为A(n,2)
        for event1, event2 in permutations(ecb_star_events, r=2):
            tid_seq1 = ecb_star_events[event1] # 事件的t_id序列
            tid_seq2 = ecb_star_events[event2]

            sen_s = e2c.get_sentence_number(tid_seq1, all_token) # 事件所在句的sentence_num
            sen_t = e2c.get_sentence_number(tid_seq2, all_token)
 
            # 忽略特殊情况:
            # 1. 两个事件位于不同句子 
            # 2. 在同一句子内, 事件一在事件二之后出现
            if sen_s!=sen_t or int(tid_seq1.split('_')[0])>=int(tid_seq2.split('_')[0]): 
                count_drop_causal['跨句'] += int((tid_seq1, tid_seq2) in ep_rel and sen_s!=sen_t)
                continue
                
            # 获取事件关系
            rel = ep_rel.get((tid_seq1, tid_seq2), 'NULL')
            # 记录因果事件对的数量
            count_event_pair['causal'] += int(rel!='NULL')
            count_event_pair['total'] += 1

            sentence = e2c.nth_sentence(sen_s, all_token) # 句子的text列表
            sen_str = ' '.join(sentence).replace('_',' ') # 句子的text序列

            span1 = e2c.get_number_list(tid_seq1, all_token) # 事件的number列表
            span2 = e2c.get_number_list(tid_seq2, all_token)

            event_mention1 = [sentence[i] for i in span1] # 事件的text列表
            event_mention2 = [sentence[i] for i in span2]

            event_str1 = '_'.join(event_mention1).lower() # 事件的text序列
            event_str2 = '_'.join(event_mention2).lower()
            if ds=='CTB':
                event_str1 = event_str1.replace('-','_')
                event_str2 = event_str2.replace('-','_')

            # 忽略该句中重复的事件对 (重复是指两个事件对中e1和e2的event_mention分别相同)
            ep_sen = (event_str1, event_str2, sen_s)
            if ep_sen in ep_sen_rel and rel=='NULL': continue
            ep_sen_rel[ep_sen].append(rel)

            zero_concept1 = MATCH_DICT[event_str1] # 事件的0-hop概念
            zero_concept2 = MATCH_DICT[event_str2]

            # 构造两个事件的描述性知识图谱
            rels1, objects1, graph1 = build_graph(subj2rel_obj, zero_concept1, cpnet_graph)
            rels2, objects2, graph2 = build_graph(subj2rel_obj, zero_concept2, cpnet_graph)

            # 获取0-hop与1-hop概念的BERT分词后的token_id列表(用于LSIN架构的模型)
            objects1_ids = tokenize(tokenizer, [zero_concept1] + objects1)
            objects2_ids = tokenize(tokenizer, [zero_concept2] + objects2)

            # =================================================================
            # 利用spacy, 根据text列表生成句法树, 选出两个事件的最短依存路径
            # sdp全称为Shortest Dependency Path, 即最短依存路径
            # =================================================================
            spacy_sent = nlp(' '.join(sentence)) 

            edges = []
            for token in spacy_sent:
                for child in token.children:
                    edges.append((str(token.i), str(child.i)))
            graph = nx.Graph(edges)

            index_1, index_2 = span1[0], span2[0] # 以事件首个token的位置代表事件的位置
            try:
                sdp_path = nx.shortest_path(graph, str(index_1), str(index_2))
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                sdp_path = [index_1, index_2]
            finally:
                sdp_path = [int(idx) for idx in sdp_path]
            # =================================================================

            # 获取两个事件的0-hop概念在ConceptNet中的最短路径 (关系路径)
            try:
                rel_path = nx.shortest_path(cpnet_graph, zero_concept1, zero_concept2)
                # 若两个事件的mention相同, 则人为设置路径
                if zero_concept1==zero_concept2: 
                    rel_path = [zero_concept1, zero_concept2]
            except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                rel_path = [zero_concept1, 'UNK', zero_concept2]
            # 关系路径中每个概念的token_id列表 (2维)
            rel_path_ids = []
            for node in rel_path:
                node = node.replace('_',' ')
                rel_path_ids.append(tokenizer(node)['input_ids'][1:-1])

            # 获取BERT分词后句子的token_id列表
            sen_bert = ['[CLS]'] + sentence + ['[SEP]']
            sen_ids = []
 
            span1 = [x+1 for x in span1] # 因为[CLS]标志符的存在, number列表所有值加一
            span2 = [x+1 for x in span2]
            sdp_path = [x+1 for x in sdp_path]

            ids_span1 = [] # 事件内所有token对应的token_id在sentenve_vec中的位置
            ids_span2 = []
            sdp_path_ids_span = [] # sdp_path内所有token对应的token_id在sentence_vec中的位置

            for i, w in enumerate(sen_bert):
                tokens = tokenizer.tokenize(w) if w not in ("[CLS]", "[SEP]") else [w]
                ids = tokenizer.convert_tokens_to_ids(tokens)
                ids_span_tmp = list(range(len(sen_ids), len(sen_ids) + len(ids)))
                if i in span1:
                    ids_span1.extend(ids_span_tmp)
                if i in span2:
                    ids_span2.extend(ids_span_tmp)
                if i in sdp_path:
                    sdp_path_ids_span.extend(ids_span_tmp)
                sen_ids.extend(ids)

            item = [
                doc_name, sen_ids, sen_ids, ids_span1, ids_span2, 
                rel, graph1, graph2, objects1_ids, objects2_ids,
                sdp_path_ids_span, rel_path_ids
            ]
            data_set.append(item)

            key = get_key(sen_ids, ids_span1, ids_span2)
            ids_dict[key].add(rel)

    print('事件对统计:', count_event_pair)
    print('句外因果事件对数量:', count_drop_causal)
    print('ids_dict键数量:', len(ids_dict))

    # 避免既存在NULL, 又存在因果关系
    for k, v in ids_dict.items():
        if 'NULL' in v and len(v)>1:
            v.remove('NULL')
            ids_dict[k] = v

    new_data = filter(ids_dict, data_set)
    count_causal = 0
    for item in new_data:
        count_causal += int(item[5]!='NULL')
    print('new_data数据:')
    print('全部事件对数量:', len(new_data))
    print('因果事件对数量:', count_causal)

    if save_mode:
        with open(save_path, 'wb') as f:
            pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)