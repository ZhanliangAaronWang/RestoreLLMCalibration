import torch
import torch.nn.functional as F
def not_max_target_confidence(group_accuracy, group_max_confidence, group_not_max_confidence, device):
    if group_accuracy == 1.0:
        alpha = 0
        gamma = 1
        beta  = 0
    else:
        constant = torch.log(torch.ones(group_max_confidence.size(dim=0))*3).unsqueeze(-1)
        second_max_confidence = group_not_max_confidence[:, 0].unsqueeze(-1)
        gamma = torch.div(constant.to(device), second_max_confidence) * (1/(1-group_accuracy))
        T = F.tanh(gamma * group_not_max_confidence).sum(dim=-1).unsqueeze(-1)
        alpha = (1 - group_accuracy)/ (T + 3)
        beta = alpha
    return alpha * F.tanh(gamma * group_not_max_confidence) + beta
