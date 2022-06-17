#===================================================
# 图卷积神经网络相关模型
# 根据LSIN中的计算公式进行复现
#===================================================

import torch
import torch.nn as nn


class GraphConvLayer(nn.Module):
    """GCNs中的单层, 计算方式遵循LSIN论文"""

    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.relu = nn.ReLU()

    def forward(self, input, adj):
        support = self.linear(input)
        output = torch.bmm(adj, support)
        output = self.relu(output + self.bias)
        return output

class GraphConvNetworks(nn.Module):
    """GCNs, 包含多个GCLayer"""

    def __init__(self):
        super(GraphConvNetworks, self).__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(768, 200)
        ])
    
    def forward(self, bert_enc, adj):

        # # 获取知识图谱中每个节点的BERT编码(用于循环初始化)
        obj_enc = bert_enc
        
        # 依次使用ModuleList中的GCLayer进行前向传播
        for layer in self.layers:
            obj_enc = layer(obj_enc, adj)

        # 获取0-hop节点的GCNs编码
        obj_enc = obj_enc[:, 0, :]

        return obj_enc


class DenseConnectedGCNs(nn.Module):
    """ DCGCNs模块, 用于关系性知识图谱构建"""

    def __init__(self, input_dim, output_dim, layer_num, dropout=0.3):
        super(DenseConnectedGCNs, self).__init__()

        # 确保输出维度可以被层数整除
        assert output_dim % layer_num==0

        self.hidden_dim = output_dim // layer_num
        self.layers = nn.ModuleList([
            GraphConvLayer(input_dim+self.hidden_dim*i, self.hidden_dim) for i in range(layer_num)
        ])
        self.output_layer = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bert_enc, adj):

        # 获取知识图谱中每个节点的BERT编码
        v_list = []
        layer_input = [bert_enc]

        # 依次使用ModuleList中的GCLayer进行前向传播
        for layer in self.layers:
            input = torch.cat(layer_input, dim=-1)
            v = layer(input, adj)
            layer_input.append(v)
            v_list.append(self.dropout(v)) 

        # 连接所有层的编码(初始BERT编码除外)作为最终编码
        v_all_layer = torch.cat(v_list, dim=-1)
        
        output = self.output_layer(v_all_layer)
        return output
