import torch
from torch import nn
from rouge import Rouge
import numpy as np
stoplist = set(['.',',', '...', '..'])

def get_reward(pred_start, pred_end, start, end, context):
    rouge = Rouge()
    p1 = pred_start.tolist()
    p2 = pred_end.tolist()
    l1 = start.tolist()
    l2 = end.tolist()
    scores = np.zeros(len(p1))
    for i in range(0, len(p1)):
        if p1[i] > p2[i]:
            pred_span = 'NO-ANSWER-FOUND'
        else:
            pred_span = ' '.join(context[i][p1[i]:p2[i]+1])
        if pred_span in stoplist:
            pred_span = 'NO-ANSWER-FOUND'
        gold_span = ' '.join(context[i][start[i]:end[i]+1])
        scores[i] = rouge.get_scores(pred_span, gold_span)[0]['rouge-l']['f']
    return torch.as_tensor(scores, dtype=torch.float32)

class DCRLLoss(nn.Module):
    def __init__(self, k):
        super(DCRLLoss, self).__init__()
        self.k = k
        self.loss = nn.NLLLoss(reduction='none')

    def forward(self, start_prob, end_prob, start, end, context):
        _, greedy_start = torch.max(start_prob, dim=1)
        _, greedy_end = torch.max(end_prob, dim=1)
        greedy_reward = get_reward(greedy_start, greedy_end, start, end, context)
        greedy_reward = greedy_reward.to(start_prob.device)
        #print (greedy_reward)
        kbest_prob_start, kbest_start = torch.topk(start_prob, self.k, dim=1)
        kbest_prob_end, kbest_end = torch.topk(end_prob, self.k, dim=1)
        indice_start = torch.multinomial(kbest_prob_start, 1)
        indice_end = torch.multinomial(kbest_prob_end, 1)
        sample_start = torch.gather(kbest_start, 1, indice_start).squeeze()
        sample_end = torch.gather(kbest_end, 1, indice_end).squeeze()
        sample_reward = get_reward(sample_start, sample_end, start, end, context)
        sample_reward = sample_reward.to(start_prob.device)
        #print (sample_reward)
        greedy_better = greedy_reward - sample_reward
        greedy_better = torch.clamp(greedy_better, 0., 1e7)
        sample_better = sample_reward - greedy_reward
        sample_better = torch.clamp(sample_reward, 0., 1e7)
        #print (greedy_better)
        #print (sample_better)
        greedy_loss_start = self.loss(start_prob, greedy_start)
        greedy_loss_end = self.loss(end_prob, greedy_end)
        sample_loss_start = self.loss(start_prob, sample_start)
        sample_loss_end = self.loss(end_prob, sample_end)
        total_loss = greedy_better * (greedy_loss_start + greedy_loss_end) + sample_better * (sample_loss_start + sample_loss_end)
        return torch.mean(total_loss)






