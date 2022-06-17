import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.nn.modules.module import Module
from transformers import BertModel

from config import CONF

DEVICE = torch.device(f'cuda:{CONF["cuda_index"]}')

class GraphConvolution(Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Linear(in_features, out_features, bias=False)
        xavier_uniform_(self.weight.weight)
        self.activations = nn.ReLU()

    def forward(self, input, adj):
        support = self.weight(input)
        # print(support.size())
        # print(adj.size())
        output = torch.bmm(adj, support)
        output = self.activations(output)
        return output
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, mem_dim, layers, dropout, device, self_loop = False):
        super(GraphConvLayer, self).__init__()
        self.mem_dim = mem_dim
        self.layers = layers
        self.head_dim = self.mem_dim // self.layers
        self.gcn_drop = dropout

        # linear transformation
        self.linear_output = nn.Linear(self.mem_dim, self.mem_dim)

        # dcgcn block
        self.weight_list = nn.ModuleList()
        for i in range(self.layers):
            self.weight_list.append(nn.Linear((self.mem_dim + self.head_dim * i), self.head_dim))

        self.weight_list = self.weight_list.to(DEVICE)
        self.linear_output = self.linear_output.to(DEVICE)
        self.self_loop = self_loop


    def forward(self, adj, gcn_inputs):
        # gcn layer
        denom = adj.sum(2).unsqueeze(2) + 1
        # print(adj.size())
        # print(gcn_inputs.size())

        outputs = gcn_inputs
        cache_list = [outputs]
        output_list = []

        for l in range(self.layers):
            Ax = adj.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            if self.self_loop:
                AxW = AxW  + self.weight_list[l](outputs)  # self loop
            else:
                AxW = AxW

            AxW = AxW / denom
            gAxW = F.relu(AxW)
            cache_list.append(gAxW)
            outputs = torch.cat(cache_list, dim=2)
            output_list.append(self.gcn_drop(gAxW))

        gcn_outputs = torch.cat(output_list, dim=2)
        gcn_outputs = gcn_outputs + gcn_inputs

        out = self.linear_output(gcn_outputs)

        return out

class StructInduction(nn.Module):
    def __init__(self, sem_dim_size, device):
        super(StructInduction, self).__init__()
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = self.sem_dim_size

        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tp_linear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)

        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tc_linear.weight)
        nn.init.constant_(self.tc_linear.bias, 0)

        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fi_linear.weight)

        self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.bilinear.weight)

        self.exparam = nn.Parameter(torch.Tensor(1, 1, self.sem_dim_size))
        torch.nn.init.xavier_uniform_(self.exparam)

        self.fzlinear = nn.Linear(3 * self.sem_dim_size, 2*self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.fzlinear.weight)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input, debug=False):  # batch*sent * token * hidden

        batch_size, token_size, dim_size = input.size()
        if token_size<=1:
            print(token_size)
        assert token_size>1

        """STEP1: Calculating Attention Matrix"""
        tp = torch.tanh(self.tp_linear(input))  # b*s, token, h1
        tc = torch.tanh(self.tc_linear(input))  # b*s, token, h1
        tp = tp.unsqueeze(2).expand(tp.size(0), tp.size(1), tp.size(1), tp.size(2)).contiguous()
        tc = tc.unsqueeze(2).expand(tc.size(0), tc.size(1), tc.size(1), tc.size(2)).contiguous()

        
        f_ij = self.bilinear(tp, tc).squeeze(-1)  # b*s, token , token
        f_i = torch.exp(self.fi_linear(input)).squeeze()  # b*s, token

        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).to(DEVICE)
        A_ij = torch.exp(f_ij) * mask

        """STEP: Incude Latent Structure"""
        tmp = torch.sum(A_ij, dim=1)  # nan: dimension
        # res = torch.zeros(batch_size, token_size, token_size).to(DEVICE)
        # # tmp = torch.stack([torch.diag(t) for t in tmp])
        # res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        res = tmp.unsqueeze(1).expand_as(A_ij) * torch.eye(token_size).to(DEVICE)
        L_ij = -A_ij + res  # A_ij has 0s as diagonals

        L_ij_bar = L_ij
        L_ij_bar[:, 0, :] = f_i

        LLinv = torch.inverse(L_ij_bar)

        d0 = f_i * LLinv[:, :, 0]

        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1, 2) * LLinv_diag).transpose(1, 2)
        tmp2 = A_ij * LLinv.transpose(1, 2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size - 1)
        temp22 = torch.ones(batch_size, token_size - 1, token_size)

        mask1 = torch.cat([temp11, temp12], 2).to(DEVICE)
        mask2 = torch.cat([temp21, temp22], 1).to(DEVICE)

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim=1)
        df = d.transpose(1, 2)

        return df

class DynamicReasoner(nn.Module):
    def __init__(self, hidden_size, gcn_layer, dropout_gcn, device):
        super(DynamicReasoner, self).__init__()
        self.hidden_size = hidden_size
        self.gcn_layer = gcn_layer
        self.dropout_gcn = dropout_gcn
        self.struc_att = StructInduction(hidden_size, device)
        self.gcn = GraphConvLayer(hidden_size, self.gcn_layer, self.dropout_gcn, device, self_loop=True)
    def forward(self, input):
        att = self.struc_att(input)
        output = self.gcn(att[:, :, 1:], input)
        return output

class BertCausalModel(nn.Module):

    def __init__(self, device, bert_grad=True, dk=False, rk=False, hn_list=[]):
        super(BertCausalModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(0.4)
        self.dim = 200
        self.dropout_gcn = nn.Dropout(0.3)
        self.trans_dim = nn.Linear(768, self.dim)

        self.reasoner_layer_first = 2
        self.reasoner_layer_second = 2
        self.reasoner = nn.ModuleList()
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_first, self.dropout_gcn, device))
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_second, self.dropout_gcn, device))
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_second, self.dropout_gcn, device))
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_second, self.dropout_gcn, device))
        self.gcn = GraphConvolution(768, self.dim)

        self.bert_grad = bert_grad
        self.dk = dk
        self.rk = rk
        self.MLP_dim_list = [768*3 + 200*2*(int(dk)+int(rk))] + hn_list + [2]
        self.fc = TopMLP(self.MLP_dim_list)


    def forward(self, sentences_s, mask_s, sentences_t, mask_t, event1, event1_mask, event2, event2_mask, graph1_adj,
                graph2_adj, object1_vec, object2_vec, mask_obj1, mask_obj2, sdp_path, sdp_path_concept, mask_sdp_path_concept):

        enc_s = self.bert(sentences_s, attention_mask = mask_s)
        enc_t = self.bert(sentences_t, attention_mask=mask_t)

        hidden_enc_s = enc_s[0]
        hidden_enc_t = enc_t[0]

        event1 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc_s, event1)], dim=0)
        event2 = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc_t, event2)], dim=0)

        m1 = event1_mask.unsqueeze(-1).expand_as(event1).float()
        m2 = event2_mask.unsqueeze(-1).expand_as(event2).float()

        event1 = event1 * m1
        event2 = event2 * m2

        opt1 = torch.sum(event1, dim=1)
        opt2 = torch.sum(event2, dim=1)

        opt = torch.cat((enc_s[1], opt1, opt2), 1)

        # latent graph
        # node_rep = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(hidden_enc_s, sdp_path)], dim=0)
        # node_rep = self.trans_dim(node_rep)
        # output = node_rep
        # for i in range(len(self.reasoner)):
        #     output = self.reasoner[i](output)
        # e1 = output[:, 0, :].view(output.size()[0], output.size()[2])
        # e2 = output[:, 1, :].view(output.size()[0], output.size()[2])

        if self.dk:
            with torch.set_grad_enabled(self.bert_grad):
                enc_obj1 = self.bert(object1_vec.view(-1, object1_vec.size()[2]), attention_mask=mask_obj1)[1].view(-1,object1_vec.size()[1], 768)
                enc_obj2 = self.bert(object2_vec.view(-1, object2_vec.size()[2]), attention_mask=mask_obj2)[1].view(-1,object2_vec.size()[1], 768)
            e1_graph = self.gcn(enc_obj1, graph1_adj)[:, 0, :].view(-1, self.dim)
            e2_graph = self.gcn(enc_obj2, graph2_adj)[:, 0, :].view(-1, self.dim)

        if self.rk:
            with torch.set_grad_enabled(self.bert_grad):
                enc_sdp_path_concept = self.bert(sdp_path_concept.view(-1, sdp_path_concept.size()[2]), attention_mask=mask_sdp_path_concept)[1].view(-1,sdp_path_concept.size()[1],768)
            enc_sdp_path_concept = self.trans_dim(enc_sdp_path_concept)
            output = enc_sdp_path_concept
            if self.dk:
                output[:, 0, :] = e1_graph
                output[:, -1, :] = e2_graph
            for i in range(len(self.reasoner)):
                output = self.reasoner[i](output)
            e1_concept = output[:, 0, :].view(output.size()[0], output.size()[2])
            e2_concept = output[:, -1, :].view(output.size()[0], output.size()[2])

        if self.dk and self.rk: # Bert+DK+RK
            opt = torch.cat((opt, e1_graph, e2_graph, e1_concept, e2_concept), 1)
        elif self.dk: # Bert+DK
            opt = torch.cat((opt, e1_graph, e2_graph), 1)
        elif self.rk: # Bert+RK
            opt = torch.cat((opt, e1_concept, e2_concept), dim=1)

        opt = self.fc(opt)
        return opt


# ==================================================================================================
# ==================================================================================================
# ==================================================================================================

class TopMLP(nn.Module):
    
    def __init__(self, hn_list):
        super(TopMLP, self).__init__()
        self.layers = nn.ModuleList([])
        for in_num, out_num in zip(hn_list[:-1], hn_list[1:]):
            self.layers.append(nn.Linear(in_num, out_num))
    
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input

class GCNs(nn.Module):
    def __init__(self):
        super(GCNs, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConvolution(768, 200))
        self.layers.append(GraphConvolution(200, 200))
    
    def forward(self, input, adj):
        output = input
        for layer in self.layers:
            output = layer(output, adj)
        return output[:, 0, :]

class Reasoner(nn.Module):
    """根据模型源代码搭建出的DK模块编码器"""
    
    def __init__(self, bert):
        super(Reasoner, self).__init__()
        self.bert = bert
        self.dropout_gcn = nn.Dropout(0.3)
        self.dim = 200
        self.trans_dim = nn.Linear(768, self.dim)

        self.reasoner_layer_first = 2
        self.reasoner_layer_second = 2
        self.reasoner = nn.ModuleList()
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_first, self.dropout_gcn, DEVICE))
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_second, self.dropout_gcn, DEVICE))
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_second, self.dropout_gcn, DEVICE))
        self.reasoner.append(DynamicReasoner(self.dim, self.reasoner_layer_second, self.dropout_gcn, DEVICE))

    def forward(self, sdp_path_concept, mask_sdp_path_concept):
        enc_sdp_path_concept = self.bert(sdp_path_concept.view(-1, sdp_path_concept.size()[2]), attention_mask=mask_sdp_path_concept)[1].view(-1,sdp_path_concept.size()[1],768)
        enc_sdp_path_concept = self.trans_dim(enc_sdp_path_concept)
        output = enc_sdp_path_concept
        for i in range(len(self.reasoner)):
            output = self.reasoner[i](output)
        e1_concept = output[:, 0, :].view(output.size()[0], output.size()[2])
        e2_concept = output[:, -1, :].view(output.size()[0], output.size()[2])

        return torch.cat([e1_concept, e2_concept], dim=-1)