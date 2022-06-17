import torch
import torch.nn as nn
from transformers import BertModel
from transformers import logging

from config import CONF
from models.attention import ScaledDotProductAttention as SDPAttention
from models.gat import GAT
from models.gcn import *
from models.lsr import *
from models.aggcn import AGGCN
from models.LSIN import Reasoner, GCNs
from util.mask import *

logging.set_verbosity_error()

def get_bert_enc(bert_model, obj_ids, obj_mask):
    """
    根据概念的token_ids与mask获取概念的BERT编码

    :param obj_ids: 3维, (batch_size, object_num, token_num)
    :param obj_mask: 2维, (batch_size*object_num, token_num)
    :return obj_enc: 3维, (batch_size, object_num, bert_dim)
    """

    with torch.no_grad():
        bert_output = bert_model(obj_ids.view(-1, obj_ids.size(-1)), obj_mask)
        obj_enc = bert_output[1].view(-1, obj_ids.size(1), 768)
        return obj_enc

    # bert_model.train()
    # bert_output = bert_model(obj_ids.view(-1, obj_ids.size(-1)), obj_mask)
    # obj_enc = bert_output[1].view(-1, obj_ids.size(1), 768)
    # return obj_enc
        

BERT_PATH = CONF.get('bert_path', 'bert-base-uncased')

# ==================================
# 基础模型
# ==================================
class ContextEncoder(nn.Module):
    """基础BERT模型, 仅获取[CLS],e1,e2的BERT表示"""

    def __init__(self, bert_model):
        super(ContextEncoder, self).__init__()
        self.model = bert_model

    def forward(self, input_ids, mask, event1, event1_mask, event2, event2_mask):

        # 使用Transformers中的BERT模型进行编码
        # self.model.train()
        enc = self.model(input_ids, attention_mask=mask)

        # 获取句子的BERT编码
        hidden_enc = enc[0]  # 每个token_id对应的编码结果, 即enc_s['last_hidden_state]
        enc_cls = enc[1]  # [CLS]编码的进一步处理结果

        # 获取事件的BERT编码
        enc_event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc, event1)], dim=0)
        enc_event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc, event2)], dim=0)

        # 调整事件掩码维度
        event1_mask = event1_mask.unsqueeze(-1).float()
        event2_mask = event2_mask.unsqueeze(-1).float()

        # 提取事件有效的BERT编码
        enc_event1 = torch.mean(enc_event1*event1_mask, dim=1)
        enc_event2 = torch.mean(enc_event2*event2_mask, dim=1)
        # enc_event1 = torch.sum(enc_event1*event1_mask, dim=1) / torch.sum(event1_mask, dim=1)
        # enc_event2 = torch.sum(enc_event2*event2_mask, dim=1) / torch.sum(event2_mask, dim=1)

        # 获取并返回最终的上下文表示
        enc_context = torch.cat([enc_cls, enc_event1, enc_event2], dim=1)
        return enc_context

# =====================================
# 由各基础模型组成的Module, 便于Model调用
# =====================================
class CEModule(nn.Module):
    """上下文编码模块, 输入句子与事件对, 输出上下文编码"""

    def __init__(self, mask=False, bert=None):
        super(CEModule, self).__init__()
        self.bert = bert if bert else BertModel.from_pretrained(BERT_PATH)
        self.encoder = ContextEncoder(self.bert)
        self.mask = mask

    def forward(self, s_ids, s_mask, e1, e1_mask, e2, e2_mask):

        # Mask处理
        if self.mask:
            s_ids, s_mask, e1, e1_mask, e2, e2_mask = mention_one_mask(s_ids, s_mask, e1, e1_mask, e2, e2_mask)
        # 获取上下文编码
        ce_rep = self.encoder(s_ids, s_mask, e1, e1_mask, e2, e2_mask)
        # 返回编码结果
        return ce_rep
    
class DKModule(nn.Module):
    """描述性知识编码模块, 输入知识图谱及1-hop概念, 输出DK编码"""

    def __init__(self, attn=False, aggcn=False, gat=False, bert=None):
        super(DKModule, self).__init__()
        self.bert = bert if bert else BertModel.from_pretrained(BERT_PATH)
        self.attn = attn # 是否使用Attention模块代替GCN模块
        self.aggcn = aggcn # 是否使用AGGCN模块代替GCN模块
        self.gat = gat # 是否使用GAT模块
        self.dk_encoder = None
        if attn:
            self.dk_encoder = SDPAttention(200, 768, 768, 8, 0.5) 
        elif aggcn:
            self.dk_encoder = AGGCN(768, 200, 0.5, 2, 2, 2, 4)
        elif gat:
            self.dk_encoder = GAT(768, 200, 200, 0.3)
        else:
            # self.dk_encoder = GraphConvNetworks() 
            self.dk_encoder = GCNs()

    def forward(self, graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2):
        
        # 获取0-hop与1-hop概念的BERT编码 (3维, batch_size, object_num, 768)
        bert_enc_e1 = get_bert_enc(self.bert, object1_vec, mask_obj1)
        bert_enc_e2 = get_bert_enc(self.bert, object2_vec, mask_obj2)

        # 描述性知识图谱编码
        dk_e1, dk_e2 = [], []
        if self.attn: # 使用注意力机制
            dk_e1 = self.dk_encoder(bert_enc_e1, bert_enc_e1, bert_enc_e1)[:, 0, :]
            dk_e2 = self.dk_encoder(bert_enc_e2, bert_enc_e2, bert_enc_e2)[:, 0, :]
        elif self.aggcn: # 使用AGGCN机制
            dk_e1 = self.dk_encoder(bert_enc_e1, graph1_adj)[:, 0, :]
            dk_e2 = self.dk_encoder(bert_enc_e2, graph2_adj)[:, 0, :]
        elif self.gat: # 使用GAT机制
            b_s = graph1_adj.size(0)
            dk_e1 = torch.stack([self.dk_encoder(bert_enc_e1[i], graph1_adj[i])[0]for i in range(b_s)])
            dk_e2 = torch.stack([self.dk_encoder(bert_enc_e2[i], graph2_adj[i])[0]for i in range(b_s)])
        else: # 使用GCN机制
            dk_e1 = self.dk_encoder(bert_enc_e1, graph1_adj)
            dk_e2 = self.dk_encoder(bert_enc_e2, graph2_adj)
        dk_rep = torch.cat([dk_e1, dk_e2], dim=-1)

        return dk_rep

class RKModule(nn.Module):
    """上下文编码关系型知识编码"""

    def __init__(self, attn=False, aggcn=False, bert=None):
        super(RKModule, self).__init__()
        self.bert = bert if bert else BertModel.from_pretrained(BERT_PATH)
        self.attn = attn
        self.aggcn = aggcn
        self.rk_encoder = None
        # self.linear = nn.Linear(768, 200)
        if attn:    self.rk_encoder = SDPAttention(200, 768, 768, 8, 0.5)
        elif aggcn: self.rk_encoder = AGGCN(768, 200, 0.5, 2, 8, 2, 4)
        else:       self.rk_encoder = ReasonModule()
    def forward(self, path_concept_ids, path_len, path_concept_mask, e1_dk_enc=None, e2_dk_enc=None):

        # 获取rel_path各节点的BERT编码
        batch_size, node_num, _ = path_concept_ids.size()
        bert_enc = get_bert_enc(self.bert, path_concept_ids, path_concept_mask)
        bert_enc = bert_enc.view(batch_size, node_num, bert_enc.size(-1))

        # 关系型知识图谱编码
        rk_enc = []
        if self.attn: # 使用自注意力机制
            attn_mask = get_mask(path_len).unsqueeze(1).to(bert_enc.device)
            rk_enc = self.rk_encoder(bert_enc, bert_enc, bert_enc, attn_mask)
        elif self.aggcn:
            adj = get_adj(path_len).to(bert_enc.device)
            rk_enc = self.rk_encoder(bert_enc, adj, e1_dk_enc, e2_dk_enc)
        else: # 使用StucAttn以及DCGCN
            rk_enc = self.rk_encoder(bert_enc)
        rk_e1 = rk_enc[:, 0, :]
        rk_e2 = torch.cat([rk_enc[i, l-1, :].unsqueeze(0) for i, l in enumerate(path_len)], dim=0)

        # bert_enc = self.bert(path_concept_ids, path_concept_mask)
        # # rk_rep = bert_enc[1]
        # rk_e1 = bert_enc[0][:, 0, :]
        # rk_e2 = torch.cat([bert_enc[0][i, l-1, :].unsqueeze(0) for i, l in enumerate(path_len)], dim=0)
        # rk_e1 = self.linear(rk_e1)
        # rk_e2 = self.linear(rk_e2)
        rk_rep = torch.cat([rk_e1, rk_e2], dim=-1)

        return rk_rep

class TopMLP(nn.Module):
    
    def __init__(self, hn_list=None):
        super(TopMLP, self).__init__()
        self.layers = nn.ModuleList([])
        for in_num, out_num in zip(hn_list[:-1], hn_list[1:]):
            self.layers.append(nn.Linear(in_num, out_num))
    
    def forward(self, input):
        for layer in self.layers:
            x = input if layer.in_features==layer.out_features else 0
            input = layer(input) + x
        return input

class DenseNN(nn.Module):
    
    def __init__(self, hn_list=None):
        super(DenseNN, self).__init__()
        self.layers = nn.ModuleList()
        self.input_list = []
        self.dim_sum = 0
        for in_num, out_num in zip(hn_list[:-1], hn_list[1:]):
            linear = nn.Linear(self.dim_sum+in_num, out_num)
            self.layers.append(linear)
            self.dim_sum += in_num
    
    def forward(self, input):
        self.input_list = [input]
        for layer in self.layers:
            input = torch.cat(self.input_list, dim=-1)
            output = layer(input)
            self.input_list.append(output)
        return output

class TopAttn(nn.Module):
    
    def __init__(self, hn_list=[]):
        super(TopAttn, self).__init__()
        self.attn = SDPAttention(100, 200, 200, 8, 0.5)
        self.linear_CE = nn.Linear(768, 200)
        self.mlp = TopMLP([700]+hn_list+[2])

    def forward(self, input):

        assert input.size(-1)==3104

        h_cls, h_e1, h_e2 = input[:, :2304].split(768, -1)
        dk_e1, dk_e2, rk_e1, rk_e2 = input[:, 2304:].split(200, -1)
        new_h_cls = self.linear_CE(h_cls)
        new_h_e1 = self.linear_CE(h_e1)
        new_h_e2 = self.linear_CE(h_e2)
        
        rep_list = [new_h_cls, new_h_e1, new_h_e2, dk_e1, dk_e2, rk_e1, rk_e2]
        attn_in = torch.stack(rep_list, dim=1)
        attn_out = self.attn(attn_in, attn_in, attn_in)
        mlp_in = attn_out.reshape(input.size(0), -1)
        mlp_out = self.mlp(mlp_in)

        return mlp_out

class MixedModel(nn.Module):
    """混合模型, 可将各个模块自由组合"""

    def __init__(self, bert=True, mask=False, dk=False, attn=False, aggcn=False, gat=False, rk=False, rk_attn=False, rk_aggcn=False, hn_list=[]):
        super(MixedModel, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_PATH) # 公用BERT模型
        self.ce_bert = CEModule(False, self.bert) if bert else None # 上下文编码模块
        self.ce_mask = CEModule(True, self.bert) if mask else None# 上下文编码模块(含mask处理)
        self.dk_module = DKModule(attn, aggcn, gat, self.bert) if dk else None # DK编码模块
        self.rk_module = RKModule(rk_attn, rk_aggcn, self.bert) if rk else None # RK编码模块
        self.MLP_dim_list = [self.get_rep_len()] + hn_list + [2]
        self.top_mlp = TopMLP(self.MLP_dim_list) # MLP模块
        # self.top_mlp = TopAttn()

    def get_rep_len(self):
        len = 0
        len += 768*3 if self.ce_bert else 0
        len += 768*3 if self.ce_mask else 0
        len += 200*2 if self.dk_module else 0
        len += 200*2 if self.rk_module else 0
        return len

    def forward(self, s_ids, s_mask, e1, e1_mask, e2, e2_mask, \
                      graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, \
                      path_concept_ids, path_len, path_concept_mask):
        
        # 各模块处理
        rep_list = []

        if self.ce_bert: # 上下文编码
            rep_list.append(self.ce_bert(s_ids, s_mask, e1, e1_mask, e2, e2_mask))
        if self.ce_mask: # 遮蔽编码
            rep_list.append(self.ce_mask(s_ids, s_mask, e1, e1_mask, e2, e2_mask))
        if self.dk_module: # DK编码
            rep_list.append(self.dk_module(graph1_adj, graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2))
        if self.rk_module: # RK编码
            rep_list.append(self.rk_module(path_concept_ids, path_len, path_concept_mask))
            # # 按照LSIN思路, 在RK编码阶段引入DK
            # if self.dk_module:
            #     e1_dk_enc, e2_dk_enc = rep_list[1][:, :200],rep_list[1][:, 200:], 
            #     rep_list.append(self.rk_module(path_concept_ids, path_len, path_concept_mask, e1_dk_enc, e2_dk_enc))
            # else:
            #     rep_list.append(self.rk_module(path_concept_ids, path_len, path_concept_mask))

        # 顶层MLP处理
        all_rep = torch.cat(rep_list, dim=-1)
        output = self.top_mlp(all_rep)

        return output