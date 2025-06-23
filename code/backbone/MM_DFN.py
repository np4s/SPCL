import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np, math

from .GCN import GCNII_lyc
from .encoder import Transformer_Encoder
def simple_batch_graphify(features, lengths, device):
    node_features = []
    batch_size = features.size(0)
    for j in range(batch_size):
        node_features.append(features[j, :lengths[j], :])

    node_features = torch.cat(node_features, dim=0)

    node_features = node_features.to(device)

    return node_features

class MMGatedAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, att_type='general', num_modal=3):
        super(MMGatedAttention, self).__init__()
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        self.num_modal = num_modal
        self.dropout_m = nn.ModuleList([nn.Dropout(0.5) for _ in range(num_modal)])

        self.transform_m = nn.ModuleList([nn.Linear(mem_dim, cand_dim, bias=True) for _ in range(num_modal)])
        
        self.tranfrorm_cross = nn.ModuleDict()
        for i in range(num_modal):
            for j in range(num_modal):
                if i == j: continue
                self.tranfrorm_cross[f"{i}{j}"] = nn.Linear(cand_dim * 3, cand_dim, bias=True)

    def forward(self, x, modals=None):

        hx = []
        for i in range(self.num_modal):
            hx.append(self.dropout_m[i](F.relu(self.transform_m[i](x[i])))) 
        
        h_out = []

        for i in range(self.num_modal):
            hi = []
            for j in range(self.num_modal):
                if i == j: continue
                z_ij = torch.sigmoid(self.tranfrorm_cross[f"{i}{j}"](torch.cat([hx[i], hx[j], hx[i] * hx[j]], dim=-1)))
                h_ij = z_ij * hx[i] + (1 - z_ij) * hx[j]
                hi.append(h_ij)
            h_out.append(torch.stack(hi).sum(dim=0))

        if len(h_out) == 0:
            return hx
        
        return h_out

class MM_DFN(nn.Module):

    def __init__(self, args):
        
        super(MM_DFN, self).__init__()

        self.args = args
        self.label_dict = args.dataset_label_dict[args.dataset]
        tag_size = len(self.label_dict)
        self.dropout = args.drop_rate
        self.return_feature = True
        self.modalities = args.modalities
        self.use_speaker = args.use_speaker
        self.num_speaker = args.dataset_num_speakers[args.dataset]
        
        self.embedding_dim = args.embedding_dim[args.dataset]
        self.n_layers = args.encoder_nlayers
        self.window_past = args.wp
        self.window_future = args.wf
        h_dim = args.hidden_dim

        self.modality_encoder = nn.ModuleDict()
        for m in self.modalities:
            self.modality_encoder.add_module(m, Transformer_Encoder(
                                                self.embedding_dim[m], 
                                                h_dim,
                                                self.args.encoder_nlayers,
                                                dropout=self.dropout))

        self.dropout_ = nn.Dropout(self.dropout)
        self.speaker_embeddings = nn.Embedding(self.num_speaker, h_dim)

        self.graph_net = GCNII_lyc(nfeat=h_dim, nlayers=self.n_layers, nhidden=h_dim, nclass=tag_size,
                               dropout=self.dropout, lamda=0.5, alpha=0.1, variant=True,
                               return_feature=True, use_residue=True)

        self.gated_attention = MMGatedAttention(2*h_dim, h_dim, att_type='general', num_modal=len(self.modalities))

        self.uni_fc = nn.ModuleDict({m:  nn.Linear(h_dim, tag_size) for m in self.modalities})

    def forward(self, data):
        x = data["tensor"]
        lengths = data["length"]
        spk_idx = data["speaker_tensor"]
        
        rep = {}
        for m in self.modalities:
            rep[m] = self.modality_encoder[m](x[m])

        _rep = rep
        rep = {}
        for m in self.modalities:
            rep[m] = simple_batch_graphify(_rep[m], lengths, self.args.device)

        rep_list = list(rep.values())
       
        adj = self.create_big_adj(rep_list, lengths.tolist(), self.modalities)

        features = torch.cat(rep_list, dim=0)

        features, layer_inners = self.graph_net(features, None, spk_idx, adj)

        all_length = rep_list[0].shape[0]

        rep = {}
        for j, m in enumerate(self.modalities):
            rep[m] = features[all_length*j:all_length * (j+1)]
        
        rep = self.gated_attention(list(rep.values()), self.modalities)
        
        logit = {}
        for j, m in enumerate(self.modalities):
            logit[m] = self.uni_fc[m](rep[j])
        
        joint = torch.stack([logit[m] for m in self.modalities], dim=0).sum(dim=0)

        return joint, logit
    
    def create_big_adj(self, features, dia_len, modals): 
        modal_num = len(modals)
        all_length = features[0].shape[0]
        adj = torch.zeros((modal_num*all_length, modal_num*all_length)).to(self.args.device)
        start = 0

        for i in range(len(dia_len)):
            sub_adjs = []
            for j, x in enumerate(features): 
                if j < 0:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i])) + torch.eye(dia_len[i])
                else:
                    sub_adj = torch.zeros((dia_len[i], dia_len[i]))
                    temp = x[start:start + dia_len[i]]
                    vec_length = torch.sqrt(torch.sum(temp.mul(temp), dim=1))
                    norm_temp = (temp.permute(1, 0) / vec_length)
                    cos_sim_matrix = torch.sum(torch.matmul(norm_temp.unsqueeze(2), norm_temp.unsqueeze(1)), dim=0)  # seq, seq
                    cos_sim_matrix = cos_sim_matrix * 0.99999
                    sim_matrix = 1 - torch.acos(cos_sim_matrix)/np.pi
                    sub_adj[:dia_len[i], :dia_len[i]] = sim_matrix
                sub_adjs.append(sub_adj)
            dia_idx = np.array(np.diag_indices(dia_len[i])) 
            for m in range(modal_num):
                for n in range(modal_num): 
                    m_start = start + all_length*m
                    n_start = start + all_length*n
                    if m == n: 
                        adj[m_start:m_start+dia_len[i], n_start:n_start+dia_len[i]] = sub_adjs[m] 
                    else:
                        modal1 = features[m][start:start+dia_len[i]] #length, dim
                        modal2 = features[n][start:start+dia_len[i]]
                        normed_modal1 = modal1.permute(1, 0) / torch.sqrt(torch.sum(modal1.mul(modal1), dim=1)) #dim, length
                        normed_modal2 = modal2.permute(1, 0) / torch.sqrt(torch.sum(modal2.mul(modal2), dim=1)) #dim, length
                        dia_cos_sim = torch.sum(normed_modal1.mul(normed_modal2).permute(1, 0), dim=1) #length
                        dia_cos_sim = dia_cos_sim * 0.99999
                        dia_sim = 1 - torch.acos(dia_cos_sim)/np.pi
                        idx =dia_idx.copy()
                        idx[0,:] += np.array([m_start]*dia_len[i])
                        idx[1,:] += np.array([n_start]*dia_len[i])
                        adj[idx[0], idx[1]] = dia_sim

            start += dia_len[i]
        d = adj.sum(1)
        D = torch.diag(torch.pow(d, -0.5))
        adj = D.mm(adj).mm(D)

        return adj
    
    