#===================================================
# 潜在结构增强(Latent Structure Refinement, LSR)模型
# 在LSIN中用于关系型知识图谱的感应与编码
#===================================================

import torch
import torch.nn as nn

from models.gcn import *

class StructAttention(nn.Module):
    """Structure Attention模块, 用于关系结构感应"""
    
    def __init__(self, input_dim, hidden_dim):
        super(StructAttention, self).__init__()
        self.linear_P = torch.nn.Linear(input_dim, hidden_dim, bias=True) 
        self.linear_C = torch.nn.Linear(input_dim, hidden_dim, bias=True)
        self.bilinear = torch.nn.Bilinear(hidden_dim, hidden_dim, 1, bias=False)  # 计算pw_score
        self.linear_R = torch.nn.Linear(input_dim, 1, bias=True) # 计算r_score

        # 权重初始化
        for layer in [self.linear_P, self.linear_C, self.bilinear, self.linear_R]:
            torch.nn.init.xavier_uniform_(layer.weight)
        for layer in [self.linear_P, self.linear_C, self.linear_R]:
            torch.nn.init.constant_(layer.bias, 0)

    def forward(self, input):
        """
        :param input: 关系路径的BERT表示, (batch_size, node_num, bert_dim)
        :return: 权重矩阵, (batch_size, node_num, node_num)
        """
        
        # 获取节点数量并确保数量大于1 (用于之后形状扩充)
        node_num = input.size(1)
        assert node_num>1
        # 获取input使用的计算设备 (用于统一mask使用的计算设备)
        device = input.device

        # 计算 pair-wise unnormalized score
        enc_P =  self.linear_P(input)
        enc_C = self.linear_C(input)
        new_enc_P = enc_P.unsqueeze(2).expand(-1, node_num, node_num, -1)
        new_enc_C = enc_C.unsqueeze(1).expand(-1, node_num, node_num, -1)
        pw_score = self.bilinear(new_enc_P.tanh(), new_enc_C.tanh()).squeeze()

        # 计算 root score
        r_score = self.linear_R(input)

        # 计算非负权重P
        mask_ij = (1 - torch.eye(node_num, node_num)).unsqueeze(0).to(device)
        P = mask_ij * pw_score.exp()

        # 计算拉普拉斯矩阵L以及它的变体\hat{L}
        P_row_sum = P.sum(dim=1).unsqueeze(1).expand(-1, node_num, node_num)
        L = (1-mask_ij)*P_row_sum - mask_ij*P
        L_var = L.clone()
        L_var[:, 0, :] = r_score[:, 0].exp()

        # 计算最终的权重矩阵A^r
        mask_kn_1j = torch.tensor([0]+[1]*(node_num-1)).unsqueeze(0).expand(node_num, -1).to(device)
        mask_kn_i1 = torch.tensor([0]+[1]*(node_num-1)).unsqueeze(1).expand(-1, node_num).to(device)
        L_var_inv = L_var.pinverse()
        A = mask_kn_1j*P.bmm(L_var_inv) - mask_kn_i1*P.bmm(L_var_inv.transpose(1,2))
        
        return A

class ReasonBlock(nn.Module):
    """由StructAttention和DCGCNs堆叠形成的Block, 用于关系型知识图谱进一步编码"""
    
    def __init__(self, input_dim, output_dim):
        super(ReasonBlock, self).__init__()
        self.attn = StructAttention(input_dim, input_dim)
        self.DCGCNs = DenseConnectedGCNs(input_dim, output_dim, 2)

    def forward(self, input):
        adj = self.attn(input)
        DCGCNs_enc = self.DCGCNs(input, adj)
        return DCGCNs_enc

class ReasonModule(nn.Module):
    """由多个ReansonBlock堆叠形成, 用于关系型知识图谱进一步编码"""

    def __init__(self):
        super(ReasonModule, self).__init__()
        self.layers = nn.ModuleList([
            ReasonBlock(768,200)
        ])
    
    def forward(self, input):
        """
        :param input: 关系路径的BERT嵌入, (batch_size, node_num, bert_dim)
        :return: 关系路径的编码结果, (batch_size, node_num, enc_dim)
        """

        # 依次使用ModuleList中的ReasonBlock进行前向传播
        layer_enc = input
        for layer in self.layers:
            layer_enc = layer(layer_enc)

        return layer_enc
