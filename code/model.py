import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import LateFusion, MMGCN, MultiDialogueGCN, MM_DFN, MultiBiModel


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        if args.backbone == "late_fusion":
            self.net = LateFusion(args)
        if args.backbone == "mmgcn":
            self.net = MMGCN(args)
        if args.backbone == "dialogue_gcn":
            self.net = MultiDialogueGCN(args)
        if args.backbone == "mm_dfn":
            self.net = MM_DFN(args)
        if args.backbone == "biddin":
            self.net = MultiBiModel(args)

        self.args = args
        self.modalities = args.modalities
        self.threshold = args.cl_threshold
        self.growing_factor = args.cl_growth

    def forward(self, data):
        joint, logit = self.net(data)

        prob = F.log_softmax(joint, dim=-1)
        prob_m = {
            m: F.log_softmax(logit[m], dim=-1) for m in self.modalities
        }
        with torch.no_grad():
            scores = {
                m: sum([F.softmax(logit[m], dim=1)[i][data["label_tensor"][i]]
                       for i in range(prob.size(0))])
                for m in self.modalities
            }

            min_score = min(scores.values())
            ratio = {
                m: scores[m] / min_score
                for m in self.modalities
            }

        return prob, prob_m, ratio

    def get_loss(self, data):
        joint, logit = self.net(data)

        prob = F.log_softmax(joint, dim=-1)
        prob_m = {
            m: F.log_softmax(logit[m], dim=-1) for m in self.modalities
        }

        loss = F.nll_loss(prob, data["label_tensor"], reduction='none')

        loss_m = {
            m: F.nll_loss(prob_m[m], data["label_tensor"])
            for m in self.modalities
        }

        with torch.no_grad():
            sum_len = [0] + torch.cumsum(data["length"], dim=0).tolist()
            score_dialogs = [
                torch.stack([torch.sum(torch.stack([F.softmax(logit[m], dim=1)[i][data["label_tensor"][i]]
                                                    for i in range(sum_len[j-1], sum_len[j])]))
                            for j in range(1, len(sum_len))])
                for m in self.modalities]
            score_dialogs = torch.stack(score_dialogs, dim=0).std(dim=0)
            dialogue_score = torch.zeros(prob.size(0)).to(self.args.device)

            for j in range(1, len(sum_len)):
                dialogue_score[sum_len[j-1]:sum_len[j]
                               ].fill_(score_dialogs[j-1])

            v = self.hard_regularization(dialogue_score, loss)

            batch_score = {
                m: sum([F.softmax(logit[m], dim=1)[
                       i][data["label_tensor"][i]] * v[i] for i in range(prob.size(0))])
                for m in self.modalities
            }

            min_score = min(batch_score.values())
            ratio = {
                m: batch_score[m] / min_score
                for m in self.modalities
            }
            
        loss = loss * v

        take_sample = torch.sum(v)

        return loss.mean(), ratio, take_sample, loss_m

    def hard_regularization(self, scores, loss):
        if self.args.use_cl:
            diff = 2 / (1 / scores + 1 / loss)
            v = diff <= self.threshold
        else:
            v = torch.ones_like(scores)
        return v.int()

    def increase_threshold(self):
        self.threshold *= self.growing_factor
        if self.threshold > 60:
            self.threshold = 60
