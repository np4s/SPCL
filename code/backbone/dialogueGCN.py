import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, GraphConv
import numpy as np


def batch_graphify(features, lengths, speaker_tensor, wp, wf, edge_type_to_idx, att_model, device):
    node_features, edge_index, edge_norm, edge_type = [], [], [], []
    batch_size = features.size(0)
    length_sum = 0
    edge_ind = []
    edge_index_lengths = []

    for j in range(batch_size):
        edge_ind.append(edge_perms(lengths[j].cpu().item(), wp, wf))

    edge_weights = att_model(features, lengths, edge_ind)

    for j in range(batch_size):
        cur_len = lengths[j].item()
        node_features.append(features[j, :cur_len, :])
        perms = edge_perms(cur_len, wp, wf)
        perms_rec = [(item[0] + length_sum, item[1] + length_sum)
                     for item in perms]
        length_sum += cur_len
        edge_index_lengths.append(len(perms))

        for item, item_rec in zip(perms, perms_rec):
            edge_index.append(torch.tensor([item_rec[0], item_rec[1]]))
            edge_norm.append(edge_weights[j][item[0], item[1]])
            # edge_norm.append(edge_weights[j, item[0], item[1]])

            speaker1 = speaker_tensor[j, item[0]].item()
            speaker2 = speaker_tensor[j, item[1]].item()
            if item[0] < item[1]:
                c = '0'
            else:
                c = '1'
            edge_type.append(
                edge_type_to_idx[str(speaker1) + str(speaker2) + c])

    node_features = torch.cat(node_features, dim=0).to(device)  # [E, D_g]
    edge_index = torch.stack(edge_index).t().contiguous().to(device)  # [2, E]
    edge_norm = torch.stack(edge_norm).to(device)  # [E]
    edge_type = torch.tensor(edge_type).long().to(device)  # [E]
    edge_index_lengths = torch.tensor(
        edge_index_lengths).long().to(device)  # [B]

    return node_features, edge_index, edge_norm, edge_type, edge_index_lengths


def edge_perms(length, window_past, window_future):
    """
    Method to construct the edges of a graph (a utterance) considering the past and future window.
    return: list of tuples. tuple -> (vertice(int), neighbor(int))
    """

    all_perms = set()
    array = np.arange(length)
    for j in range(length):
        perms = set()

        if window_past == -1 and window_future == -1:
            eff_array = array
        elif window_past == -1:  # use all past context
            eff_array = array[:min(length, j + window_future + 1)]
        elif window_future == -1:  # use all future context
            eff_array = array[max(0, j - window_past):]
        else:
            eff_array = array[max(0, j - window_past)                              :min(length, j + window_future + 1)]

        for item in eff_array:
            perms.add((j, item))
        all_perms = all_perms.union(perms)
    return list(all_perms)


class SeqContext(nn.Module):
    def __init__(self, u_dim, g_dim, drop_rate=0.5, nhead=1, nlayer=1):
        super(SeqContext, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=u_dim,
            nhead=nhead,
            dropout=drop_rate,
            batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=nlayer)
        self.transformer_out = torch.nn.Linear(u_dim, g_dim, bias=True)

    def forward(self, x):
        rnn_out = self.transformer_encoder(x)
        rnn_out = self.transformer_out(rnn_out)
        return rnn_out


class EdgeAtt(nn.Module):

    def __init__(self, g_dim, wp, wf, device):
        super(EdgeAtt, self).__init__()
        self.device = device
        self.wp = wp
        self.wf = wf

        self.weight = nn.Parameter(torch.zeros(
            (g_dim, g_dim)).float(), requires_grad=True)
        var = 2. / (self.weight.size(0) + self.weight.size(1))
        self.weight.data.normal_(0, var)

    def forward(self, node_features, text_len_tensor, edge_ind):
        batch_size, mx_len = node_features.size(0), node_features.size(1)
        alphas = []

        weight = self.weight.unsqueeze(0).unsqueeze(0)
        att_matrix = torch.matmul(
            weight, node_features.unsqueeze(-1)).squeeze(-1)  # [B, L, D_g]
        for i in range(batch_size):
            cur_len = text_len_tensor[i].item()
            alpha = torch.zeros((mx_len, 110)).to(self.device)
            for j in range(cur_len):
                s = j - self.wp if j - self.wp >= 0 else 0
                e = j + self.wf if j + self.wf <= cur_len - 1 else cur_len - 1
                tmp = att_matrix[i, s: e + 1, :]  # [L', D_g]
                feat = node_features[i, j]  # [D_g]
                score = torch.matmul(tmp, feat)
                probs = F.softmax(score, -1)  # [L']
                alpha[j, s: e + 1] = probs
            alphas.append(alpha)

        return alphas


class GCN(nn.Module):

    def __init__(self, g_dim, h1_dim, h2_dim, n_speakers=2):
        super(GCN, self).__init__()
        self.num_relations = 2 * n_speakers ** 2
        self.conv1 = RGCNConv(g_dim, h1_dim, self.num_relations, num_bases=30)
        self.conv2 = GraphConv(h1_dim, h2_dim)

    def forward(self, node_features, edge_index, edge_norm, edge_type):
        x = self.conv1(node_features, edge_index, edge_type)
        x = self.conv2(x, edge_index, edge_norm)

        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_size, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        # print(input_dim, hidden_size, n_classes)
        self.lin1 = nn.Linear(input_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, n_classes)
        self.dropout = nn.Dropout(dropout)
        self.nll_loss = nn.NLLLoss()

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        return x

    def get_prob(self, x):
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        log_prob = F.log_softmax(x, dim=-1)
        return log_prob

    def get_loss(self, x, labels):
        log_prob = self.get_prob(x)
        loss = self.nll_loss(log_prob, labels)
        return log_prob, loss


class DialogueGCN(nn.Module):

    def __init__(self, u_dim, g_dim, h_dim, device, tag_size, n_speakers, wp, wf, nhead, nlayers, dropout=0.1):
        super(DialogueGCN, self).__init__()

        self.wp = wp
        self.wf = wf
        self.device = device

        self.rnn = SeqContext(u_dim, g_dim, dropout, nhead, nlayers)
        self.edge_att = EdgeAtt(g_dim, self.wp, self.wf, self.device)
        self.gcn = GCN(g_dim, h_dim, h_dim, n_speakers)
        self.clf = Classifier(g_dim + h_dim, h_dim, tag_size, dropout)

        edge_type_to_idx = {}
        for j in range(n_speakers):
            for k in range(n_speakers):
                edge_type_to_idx[str(j) + str(k) + '0'] = len(edge_type_to_idx)
                edge_type_to_idx[str(j) + str(k) + '1'] = len(edge_type_to_idx)
        self.edge_type_to_idx = edge_type_to_idx

    def get_rep(self, x, length, speaker):
        node_features = self.rnn(x)  # [batch_size, mx_len, D_g]
        features, edge_index, edge_norm, edge_type, edge_index_lengths = batch_graphify(
            node_features, length, speaker, self.wp, self.wf,
            self.edge_type_to_idx, self.edge_att, self.device)

        graph_out = self.gcn(features, edge_index, edge_norm, edge_type)

        return graph_out, features

    def forward(self, x, length, speaker):
        graph_out, features = self.get_rep(x, length, speaker)
        out = self.clf(torch.cat([features, graph_out], dim=-1))

        return out

    def get_loss(self, data):
        graph_out, features = self.get_rep(data)
        loss = self.clf.get_loss(torch.cat([features, graph_out], dim=-1),
                                 data["label_tensor"], data["text_len_tensor"])

        return loss


class MultiDialogueGCN(nn.Module):

    def __init__(self, args):
        super(MultiDialogueGCN, self).__init__()

        self.args = args
        self.label_dict = args.dataset_label_dict[args.dataset]
        tag_size = len(self.label_dict)
        self.dropout = args.drop_rate
        self.modalities = args.modalities
        self.num_speaker = args.dataset_num_speakers[args.dataset]

        self.embedding_dim = args.embedding_dim[args.dataset]
        self.nlayers = args.encoder_nlayers
        self.nhead = args.trans_head
        self.window_past = args.wp
        self.window_future = args.wf

        self.g_dim = args.hidden_dim
        self.h_dim = args.hidden2_dim

        self.uni_fc = nn.ModuleDict()
        for m in self.modalities:
            self.uni_fc.add_module(m, DialogueGCN(self.embedding_dim[m], self.g_dim, self.h_dim, args.device, tag_size,
                                   self.num_speaker, self.window_past, self.window_future, self.nhead, self.nlayers, self.dropout))

    def forward(self, data):
        x = data["tensor"]
        length = data["length"]
        speaker = data["speaker_tensor"]
        logit = {}
        for m in self.modalities:
            logit[m] = self.uni_fc[m](x[m], length, speaker)

        joint = torch.stack([logit[m]
                            for m in self.modalities], dim=0).sum(dim=0)
        return joint, logit
