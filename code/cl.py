import torch
import torch.nn as nn   
import torch.nn.functional as F

class SPLLoss (nn.NLLLoss):
    def __init__(self, n_sample, args):
        super(SPLLoss, self).__init__()
        self.device = args.device
        if args.regularizer=='hard':
            self.regularizer = 'hard'
            self.threshold = args.cl_threshold
            self.growing_factor = args.cl_growth
            self.v = torch.zeros(n_sample, dtype=torch.int64).to(self.device)
            
    def forward(self, prob, labels, return_take_samp=False):
        difficulty = F.nll_loss(prob, labels, reduction='none')
        # print("threshold", self.threshold)
        if self.regularizer=='hard':
            v = self.hard_regularization(difficulty)
            take_samp = torch.sum(v)
            rate = take_samp/len(v)
            loss = difficulty * v
            if return_take_samp:
                return loss.mean(), take_samp
            return loss.mean()
    
    def increase_threshold(self):
        self.threshold *= self.growing_factor
        if self.threshold > 60:
          self.threshold = 60
        
    def hard_regularization(self, difficulty):
        v = difficulty < self.threshold
        return v.int()


class DialogSPCLLoss(nn.Module):
    def __init__(self, n_sample, args):
        super(DialogSPCLLoss, self).__init__()
        self.device = args.device
        self.n_sample = n_sample
        self.growing_factor = args.cl_growth
        self.v = torch.zeros(n_sample, dtype=torch.int64).to(self.device)
        self.threshold = args.cl_threshold
    
    def forward(self, prob, labels, scores, return_take_samp=False):
        difficulty = F.nll_loss(prob, labels, reduction='none')
        v = self.hard_regularization(difficulty, scores)
        take_samp = torch.sum(v)
        rate = take_samp/len(v)
        loss = difficulty * v
        if return_take_samp:
            return loss.mean(), take_samp
        return loss.mean()
    
    def increase_threshold(self):
        self.threshold *= self.growing_factor
        if self.threshold > 60:
          self.threshold = 60

    def hard_regularization(self, difficulty, scores):
        v = scores < self.threshold
        return v.int()

