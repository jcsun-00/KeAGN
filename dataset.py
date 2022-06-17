#===================================================
# 自定义数据集文件
#===================================================

import math
import sys
import torch
import pickle
import random
import numpy as np

from tqdm import tqdm
from allennlp.common.util import pad_sequence_to_length
from allennlp.nn.util import get_mask_from_sequence_lengths

from util.mask import mention_one_mask

class Dataset(object):
    def __init__(self, batch_size, dataset, mention_mask=False):
        super(Dataset, self).__init__()

        self.mention_mask = mention_mask # 是否包含Mask处理后的数据
        self.batch_size = batch_size
        self.batch_num = math.ceil(len(dataset)/batch_size)
        self.y_label = {
            'NULL': 0,
            'null': 0,
            'FALLING_ACTION': 1,
            'PRECONDITION': 1,
            'Coref': 1
        }

        self.construct_index(dataset)

    def __len__(self):
        return self.index_length

    def construct_index(self, dataset):
        """构建shuffle索引"""

        self.dataset = dataset
        self.index_length = len(dataset)
        self.shuffle_list = list(range(0, self.index_length))

    def shuffle(self):
        """对索引进行shuffle操作"""
        
        random.shuffle(self.shuffle_list)

    def get_tqdm(self, device, shuffle=True):
        """获取读取数据的进度条"""

        return tqdm(self.reader(device, shuffle), mininterval=2, \
                total=self.index_length // self.batch_size, \
                leave=False, file=sys.stderr, ncols=80)

    def reader(self, device, shuffle):
        cur_idx = 0
        if shuffle:
            self.shuffle()
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch_list = [self.shuffle_list[index] for index in range(cur_idx, end_index)]
            batch = [self.dataset[index] for index in batch_list]
            cur_idx = end_index
            yield self.batchify(batch, device)

    def reader_with_source_data(self, device, shuffle):
        cur_idx = 0
        if shuffle:
            self.shuffle()
        while cur_idx < self.index_length:
            end_index = min(cur_idx + self.batch_size, self.index_length)
            batch_list = [self.shuffle_list[index] for index in range(cur_idx, end_index)]
            batch = [self.dataset[index] for index in batch_list]
            cur_idx = end_index
            yield self.batchify(batch, device), batch

    def add_mask(self, batch):
        """在文本中对事件进行One-Word Mask处理"""
        
        sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, \
            data_y, graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
            sdp_path, rel_path, rel_path_len, mask_sdp_path_concept = batch
        
        # # 从结果文件中检索
        # result = match(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
        # 从原数据生成
        result = mention_one_mask(sentences_s, mask_s, event1, event1_mask, event2, event2_mask)
        new_ids, new_mask, new_e1, new_e1_mask, new_e2, new_e2_mask = result
        batch = [   
            sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, \
            data_y, graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
            sdp_path, rel_path, rel_path_len, mask_sdp_path_concept, \
            new_ids, new_mask, new_e1, new_e1_mask, new_e2, new_e2_mask
        ]
        return batch

    def batchify(self, batch, device):

        # batch中每行数据的列含义:
        # 0 doc             1 sentence_vec_s    2  sentence_vec_t
        # 3 span1_vec       4 span2_vec         5  rel
        # 6 graph1          7 graph2            8  objects1_vec
        # 9 objects2_vec    10 sdp_path_vec     11 sdp_path_concept_vec

        sentence_len = [len(tup[1]) for tup in batch] # BERT分词后, 句子中token的数量
        max_sentence_len = max(sentence_len) # 当前批次句子的最大长度

        event1_lens = [len(tup[3]) for tup in batch] # BERT分词后, 事件中token的数量
        event2_lens = [len(tup[4]) for tup in batch]

        sdp_path_len = [len(tup[10]) for tup in batch] # 当前批次每行sdp_path中token的数量
        max_sdp_path_len = max(sdp_path_len) # 当前批次sdp_path中token数量的最大值

        object1_len = [] # 当前批次中, 所有事件的0-hop和1-hop概念的token数量
        object2_len = []
        object1_len_ = [] # 当前批次中, 所有事件的0-hop和1-hop概念的token数量 (2维, 行为事件, 列为概念)
        object2_len_ = []

        for tup in batch:
            object1 = tup[8] # 事件0-hop和1-hop概念的BERT分词后的token_id列表 (2维)
            object2 = tup[9]


            object1_len_.append([len(o) for o in object1]) # 事件0-hop和1-hop概念中, 每个概念的token_id列表长度
            object2_len_.append([len(o) for o in object2])

            for i in range(len(object1)):
                object1_len.append(len(object1[i]))
                object2_len.append(len(object2[i]))

        max_object1_len = max(object1_len)
        max_object2_len = max(object2_len)

        sentences_s, sentences_t, event1, event2, data_y, object1_vec, object2_vec = [], [], [], [], [], [], []
        sdp_path, sdp_path_concept_vec = [], []
        graph1, graph2 = [], []

        for data in batch:
            sentences_s.append(data[1])
            sentences_t.append(data[2])
            event1.append(data[3])
            event2.append(data[4])
            data_y.append(self.y_label[data[5]])
            graph1.append(data[6])
            graph2.append(data[7])
            object1_vec.append(list(map(lambda x: pad_sequence_to_length(x, max_object1_len), data[8])))
            object2_vec.append(list(map(lambda x: pad_sequence_to_length(x, max_object2_len), data[9])))
            sdp_path.append(data[10])

        sentences_s = list(map(lambda x: pad_sequence_to_length(x, max_sentence_len), sentences_s))
        sentences_t = list(map(lambda x: pad_sequence_to_length(x, max_sentence_len), sentences_t))

        event1 = list(map(lambda x: pad_sequence_to_length(x, 5), event1))
        event2 = list(map(lambda x: pad_sequence_to_length(x, 5), event2))

        sdp_path = list(map(lambda x: pad_sequence_to_length(x, max_sdp_path_len), sdp_path))

        mask_sentences_s = get_mask_from_sequence_lengths(torch.tensor(sentence_len), max_sentence_len)
        mask_sentences_t = get_mask_from_sequence_lengths(torch.tensor(sentence_len), max_sentence_len)

        mask_even1 = get_mask_from_sequence_lengths(torch.tensor(event1_lens), 5)
        mask_even2 = get_mask_from_sequence_lengths(torch.tensor(event2_lens), 5)

        mask_obj1 = []
        mask_obj2 = []
        for i in range(len(object1_len_)):
            mask_obj1.append(get_mask_from_sequence_lengths(torch.tensor(object1_len_[i]), max_object1_len))
            mask_obj2.append(get_mask_from_sequence_lengths(torch.tensor(object2_len_[i]), max_object2_len))
        mask_obj1 = torch.cat(mask_obj1, dim = 0)
        mask_obj2 = torch.cat(mask_obj2, dim = 0)

        graph1, graph2 = np.array(graph1), np.array(graph2)

        # 对仅包含概念的rel_path进行处理
        rel_path_len = [len(tup[11]) for tup in batch] # 1维, 元素值为关系路径长度(即关系路径中的concept数量)
        max_rel_path_len = max(rel_path_len) # 当前批次中, 关系路径的最大长度
        max_rel_path_concept_len = max([len(concept) for data in batch for concept in data[11]])
        sdp_path_concept_vec = []
        rel_path_concept_len_ = []
        for data in batch:
            rel_path = data[11]
            # 若关系路径的长度小于该批次最大长度, 则用[0]进行填充对齐
            if len(rel_path)<max_rel_path_len:
                rel_path.extend([[0]]*(max_rel_path_len-len(rel_path)))
            # 记录每个concept的token_id列表的原始长度
            rel_path_concept_len_.append([len(o) for o in rel_path])
            # 对rel_path中的每个concept的token_id列表进行填充对齐
            sdp_path_concept_vec.append([pad_sequence_to_length(x, max_rel_path_concept_len) for x in rel_path])
        # 为概念扩充后的rel_path生成掩码
        mask_sdp_path_concept = []
        for i in range(len(batch)):
            mask_sdp_path_concept.append(get_mask_from_sequence_lengths(torch.tensor(rel_path_concept_len_[i]), max_rel_path_concept_len))
        mask_sdp_path_concept = torch.cat(mask_sdp_path_concept, dim=0)

        batch = [torch.LongTensor(sentences_s), mask_sentences_s,
                torch.LongTensor(sentences_t), mask_sentences_t,
                torch.LongTensor(event1), mask_even1,
                torch.LongTensor(event2), mask_even2,
                torch.LongTensor(data_y), 
                torch.FloatTensor(graph1), torch.FloatTensor(graph2),
                torch.LongTensor(object1_vec), torch.LongTensor(object2_vec),
                mask_obj1, mask_obj2, 
                torch.LongTensor(sdp_path),
                torch.LongTensor(sdp_path_concept_vec), 
                torch.LongTensor(rel_path_len),
                mask_sdp_path_concept]

        new_batch = self.add_mask(batch) if self.mention_mask else batch
        new_batch = [x.to(device) for x in new_batch]
        return new_batch

if __name__ == '__main__':

    # 测试打印时间
    # 不添加Mask, 设备为cuda, 用时约30s, 设备为cpu, 用时约5s
    # 添加Mask, 设备为cuda, 用时约40s, 设备为cpu, 用时约5s
    from util.train import show_time
    with open('./data/data_v4.2.pickle', 'rb') as f:
        data = pickle.load(f)
    print('Load Finished.')
    show_time()
    print('Loop Started.')
    dataset = Dataset(20, data, True)
    for batch in dataset.reader('cuda:3', True):
        show_time()
    print('Loop Finished.')