import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class SimpleAttention(nn.Module):

    def __init__(self, input_dim):
        super(SimpleAttention, self).__init__()
        self.input_dim = input_dim
        self.scalar = nn.Linear(self.input_dim, 1, bias=False)

    def forward(self, M, x=None):
        """
        M -> (seq_len, batch, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        scale = self.scalar(M)  # seq_len, batch, 1
        alpha = F.softmax(scale, dim=0).permute(1, 2, 0)  # batch, 1, seq_len
        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[
            :, 0, :]  # batch, vector

        return attn_pool, alpha


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha_ = F.softmax((torch.bmm(x_, M_)) *
                               mask.unsqueeze(1), dim=2)  # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2,
                                  keepdim=True)  # batch, 1, 1
            alpha = alpha_masked/alpha_sum  # batch, 1, 1 ; normalized
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            # batch, seqlen, cand_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(
                1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[
            :, 0, :]  # batch, mem_dim

        return attn_pool, alpha


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''

        # scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask:
            # Fills elements of self tensor with value where mask is True.
            scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, cand_dim, embed_dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.transform = nn.Linear(cand_dim, embed_dim, bias=False)
        assert embed_dim % n_heads == 0
        self.cand_dim = cand_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.dim_per_head = self.embed_dim // self.n_heads
        self.W_Q = nn.Linear(
            self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.W_K = nn.Linear(
            self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.W_V = nn.Linear(
            self.embed_dim, self.dim_per_head * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.dim_per_head,
                            self.embed_dim, bias=False)

    def forward(self, input_Q, input_K, input_V):
        # input_Q: [seq_len, batch_size, cand_dim]  or [batch_size, cand_dim]
        # input_K: [seq_len, batch_size, embed_dim]
        # input_V: [seq_len, batch_size, embed_dim]
        if len(input_Q.shape) == 2:
            input_Q = input_Q.unsqueeze(0)
        if input_Q.size(-1) != input_K.size(-1):
            # [1/seq_len, batch_size, embed_dim]
            input_Q = self.transform(input_Q)
        residual = input_Q  # [1/seq_len, batch_size, embed_dim]
        batch_size = input_Q.size(1)
        # [batch_size, n_heads, 1/seq_len, dim_per_head]
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads,
                                   self.dim_per_head).transpose(1, 2)
        # [batch_size, n_heads, seq_len, dim_per_head]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads,
                                   self.dim_per_head).transpose(1, 2)
        # [batch_size, n_heads, seq_len, dim_per_head]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads,
                                   self.dim_per_head).transpose(1, 2)

        context, attn = ScaledDotProductAttention(self.dim_per_head)(Q, K, V)
        # context: [batch_size, n_heads, 1/seq_len, dim_per_head]
        # attn: [batch_size, n_heads, 1/seq_len, seq_len]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, self.n_heads * self.dim_per_head).transpose(0, 1)  # [1/seq_len, batch_size, embed_size]
        output = self.fc(context)  # [1/seq_len, batch_size, embed_size]
        # [1/seq_len, batch_size, seq_len]
        attn = attn.reshape(attn.size(2), -1, attn.size(3))
        return nn.LayerNorm(self.embed_dim).cuda()(output + residual), attn


class SelfAttention(nn.Module):

    def __init__(self, input_dim, att_type='general'):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.att_type = att_type
        self.scalar = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, M, x=None):
        """
        now M -> (batch, seq_len, vector)
        x -> dummy argument for the compatibility with MatchingAttention
        """
        if self.att_type == 'general':
            scale = self.scalar(M)  # seq_len, batch, 1
            alpha = F.softmax(scale, dim=0).permute(
                0, 2, 1)  # batch, 1, seq_len
            attn_pool = torch.bmm(alpha, M)[:, 0, :]  # batch, vector/input_dim
        if self.att_type == 'general2':
            scale = self.scalar(M)  # seq_len, batch, 1
            alpha = F.softmax(scale, dim=0).permute(
                0, 2, 1)  # batch, 1, seq_len
            att_vec_bag = []
            for i in range(M.size()[1]):
                alp = alpha[:, :, i]
                vec = M[:, i, :]
                alp = alp.repeat(1, self.input_dim)
                att_vec = torch.mul(alp, vec)  # batch, vector/input_dim
                att_vec = att_vec + vec
                att_vec_bag.append(att_vec)
            attn_pool = torch.cat(att_vec_bag, -1)

        return attn_pool, alpha


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))
        self.dropout = nn.Dropout(dropout)
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
                0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):

        # X: seq_len, batch_size, num_hiddens
        X = X.transpose(0, 1)
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X.transpose(0, 1))


class DialogueRNNCell(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_att=100, dropout=0.5):
        super(DialogueRNNCell, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e

        self.listener_state = listener_state
        self.g_cell = nn.GRUCell(D_m+D_p, D_g)
        self.p_cell = nn.GRUCell(D_m+D_g, D_p)
        self.e_cell = nn.GRUCell(D_p, D_e)
        if listener_state:
            self.l_cell = nn.GRUCell(D_m+D_p, D_p)

        self.dropout = nn.Dropout(dropout)
        self.positional_embedding = PositionalEncoding(D_g)

        if context_attention == 'simple':
            self.attention = SimpleAttention(D_g)
        else:
            self.attention = MatchingAttention(
                D_g, D_m, D_att, context_attention)

    def rnn_cell(self, U, c_, qmask, qm_idx, q0, e0, p_cell, e_cell):
        U_c_ = torch.cat([U, c_], dim=1).unsqueeze(
            1).expand(-1, qmask.size()[1], -1)
        qs_ = p_cell(U_c_.contiguous().view(-1, self.D_m+self.D_g),
                     q0.view(-1, self.D_p)).view(U.size()[0], -1, self.D_p)
        qs_ = self.dropout(qs_)

        if self.listener_state:
            U_ = U.unsqueeze(1).expand(-1, qmask.size()
                                       [1], -1).contiguous().view(-1, self.D_m)
            ss_ = self._select_parties(qs_, qm_idx).unsqueeze(1).\
                expand(-1, qmask.size()[1], -1).contiguous().view(-1, self.D_p)
            U_ss_ = torch.cat([U_, ss_], 1)
            ql_ = self.l_cell(U_ss_, q0.view(-1, self.D_p)
                              ).view(U.size()[0], -1, self.D_p)
            ql_ = self.dropout(ql_)
        else:
            ql_ = q0
        qmask_ = qmask.unsqueeze(2)
        q_ = ql_*(1-qmask_) + qs_*qmask_
        e0 = torch.zeros(qmask.size()[0], self.D_e).type(U.type()) if e0.size()[0] == 0\
            else e0
        e_ = e_cell(self._select_parties(q_, qm_idx), e0)
        e_ = self.dropout(e_)
        return q_, e_

    def _select_parties(self, X, indices):
        q0_sel = []
        for idx, j in zip(indices, X):
            q0_sel.append(j[idx].unsqueeze(0))
        q0_sel = torch.cat(q0_sel, 0)
        return q0_sel

    def forward(self, U, qmask, g_hist, q0, e0):
        """
        U -> batch, D_m
        qmask -> batch, party
        g_hist -> t-1, batch, D_g
        q0 -> batch, party, D_p
        e0 -> batch, self.D_e
        """
        qm_idx = torch.argmax(qmask, 1)
        q0_sel = self._select_parties(q0, qm_idx)

        g_ = self.g_cell(torch.cat([U, q0_sel], dim=1),
                         torch.zeros(U.size()[0], self.D_g).type(U.type()) if g_hist.size()[0] == 0 else
                         g_hist[-1])
        g_ = self.dropout(g_)
        if g_hist.size()[0] == 0:
            c_ = torch.zeros(U.size()[0], self.D_g).type(U.type())
            alpha = None
        else:
            g_hist = self.positional_embedding(g_hist)
            c_, alpha = self.attention(g_hist, U)

        q_, e_ = self.rnn_cell(U, c_, qmask, qm_idx, q0,
                               e0, self.p_cell, self.e_cell)

        return g_, q_, e_, alpha


class DialogueRNN(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, listener_state=False,
                 context_attention='simple', D_att=100, dropout=0.5):
        super(DialogueRNN, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.dropout = nn.Dropout(dropout)

        self.dialogue_cell = DialogueRNNCell(D_m, D_g, D_p, D_e,
                                             listener_state, context_attention, D_att, dropout)

    def forward(self, U, qmask):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """

        g_hist = torch.zeros(0).type(U.type())  # 0-dimensional tensor
        q_ = torch.zeros(qmask.size()[1], qmask.size()[2],
                         self.D_p).type(U.type())  # batch, party, D_p
        e_ = torch.zeros(0).type(U.type())  # batch, D_e
        e = e_

        alpha = []
        for u_, qmask_ in zip(U, qmask):
            g_, q_, e_, alpha_ = self.dialogue_cell(u_, qmask_, g_hist, q_, e_)
            g_hist = torch.cat([g_hist, g_.unsqueeze(0)], 0)
            e = torch.cat([e, e_.unsqueeze(0)], 0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])

        return e, alpha  # [seq_len, batch, D_e]


class BiModel_single(nn.Module):

    def __init__(self, D_m, D_g, D_p, D_e, D_h,
                 n_classes=7, listener_state=False, context_attention='simple', D_att=100, dropout=0.5):
        super(BiModel_single, self).__init__()

        self.D_m = D_m
        self.D_g = D_g
        self.D_p = D_p
        self.D_e = D_e
        self.D_h = D_h
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout)
        self.dialog_rnn_f = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_att, dropout)
        self.dialog_rnn_r = DialogueRNN(D_m, D_g, D_p, D_e, listener_state,
                                        context_attention, D_att, dropout)
        self.positional_embedding1 = PositionalEncoding(D_e)
        self.positional_embedding = PositionalEncoding(D_e*2)

        self.linear = nn.Linear(2*D_e, 2*D_h)
        self.smax_fc = nn.Linear(2*D_h, n_classes)
        self.multihead_attn = MultiHeadAttention(2*D_e, 2*D_e, 4)

    def _reverse_seq(self, X, mask):
        """
        X -> seq_len, batch, dim
        mask -> batch, seq_len
        """
        X_ = X.transpose(0, 1)
        mask_sum = torch.sum(mask, 1).int()

        xfs = []
        for x, c in zip(X_, mask_sum):
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

    def forward(self, U, qmask, umask, att2=True):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions_f, alpha_f = self.dialog_rnn_f(
            U, qmask)  # seq_len, batch, D_e
        # emotions_f = self.positional_embedding1(emotions_f)
        emotions_f = self.dropout_rec(emotions_f)
        rev_U = self._reverse_seq(U, umask)
        rev_qmask = self._reverse_seq(qmask, umask)
        emotions_r, alpha_r = self.dialog_rnn_r(rev_U, rev_qmask)
        # emotions_r = self.positional_embedding1(emotions_r)
        emotions_r = self._reverse_seq(emotions_r, umask)
        emotions_r = self.dropout_rec(emotions_r)
        emotions = torch.cat([emotions_f, emotions_r], dim=-1)
        emotions = self.positional_embedding(
            emotions)  # seq_len, batch_size, De*6
        if att2:
            # MultiHeadAttention
            att_emotions, alpha = self.multihead_attn(
                emotions, emotions, emotions)
            # att_emotions : e'=[e1',e2',...,en']
            # seq_len, batch_size, Dh*3
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))

        hidden = self.dropout(hidden)
        # seq_len, batch, n_classes
        log_prob = self.smax_fc(hidden)
        log_prob = log_prob.transpose(0, 1)
        return log_prob


class MultiBiModel(nn.Module):
    def __init__(self, args):
        super(MultiBiModel, self).__init__()
        """
        hidden_dim: global context size vector
        hidden2_dim: party's state
        hidden3_dim: emotion's represent
        hidden4_dim: linear's emotion's represent
        
        """

        self.args = args
        self.label_dict = args.dataset_label_dict[args.dataset]
        tag_size = len(self.label_dict)
        self.dropout = args.drop_rate
        self.modalities = args.modalities

        self.embedding_dim = args.embedding_dim[args.dataset]
        self.device = args.device

        h_dim = args.hidden_dim
        h2_dim = args.hidden2_dim
        h3_dim = args.hidden3_dim
        h4_dim = args.hidden4_dim
        self.listener_state = args.listener_state
        self.context_attention = args.context_attention
        self.D_att = args.D_att

        self.uni_bimodel = nn.ModuleDict()
        for m in self.modalities:
            self.uni_bimodel.add_module(m, BiModel_single(self.embedding_dim[m],
                                                          h_dim, h2_dim, h3_dim, h4_dim,
                                                          n_classes=tag_size,
                                                          listener_state=self.listener_state,
                                                          context_attention=self.context_attention,
                                                          D_att=self.D_att, dropout=self.dropout))

        self.uni_fc = nn.ModuleDict(
            {m: nn.Linear(h4_dim, tag_size) for m in self.modalities})

    def forward(self, data):
        x = data["tensor"]
        qmask = F.one_hot(
            data["speaker_tensor"].transpose(0, 1)).to(self.device)
        umask = pad_sequence([torch.ones(s_len) for s_len in data["length"]]).transpose(
            0, 1).to(self.device)

        logit = {}

        for m in self.modalities:
            logit[m] = self.uni_bimodel[m](x[m].transpose(0, 1), qmask, umask)
            # print(data["label_tensor"].shape, rep.shape)
            # exit(0)
            # logit[m] = self.uni_fc[m](rep)

        logit = self.logit_flating(
            logit, data["length"], self.modalities, self.device)
        joint = torch.stack([logit[m]
                            for m in self.modalities], dim=0).sum(dim=0)
        return joint, logit

    def logit_flating(self, batches, lengths, modalities, device):

        logits = {
            m: [] for m in modalities
        }

        lengths = lengths.tolist()
        batch_size = len(lengths)

        for j in range(batch_size):
            cur_len = lengths[j]
            for m in modalities:
                logits[m].append(batches[m][j, :cur_len])

        for m in modalities:
            logits[m] = torch.cat(logits[m], dim=0).to(device)

        return logits