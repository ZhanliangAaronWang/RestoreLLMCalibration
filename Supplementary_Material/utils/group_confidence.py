import torch
import torch.nn.functional as F

def group_confidences(logits, tokenizer, upper_boundary): # The confidence distribution matrix for a batch of samples,
                                             # We need calculate the loss for each label of each sample w.r.t group accuracy
    if logits.size(dim=0) == 0:
        return torch.zeros(1, requires_grad=True)
    label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D', 'E']}
    label_logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D', 'E']]]
    probs = torch.exp(label_logits - torch.logsumexp(label_logits, dim=-1, keepdim=True))
    indices = torch.bucketize(probs, upper_boundary, right=True).squeeze()
    group_not_max_confidence = []
    group_max_confidence = []
    confidence_options = {"A":[], "B":[], "C":[], "D":[]}
    options_list = ['A', 'B', 'C', 'D']
    for k in range(1, len(upper_boundary)+1):
        for idx, option in enumerate(options_list):
            probs_option = probs[:, idx] # N x 1
            indices_option = indices[:, idx] # N x 1
            mask = (indices_option == k)
            if mask.sum().item() == 0:
                confidence_options[option].append(torch.zeros(1, requires_grad=True)[0].to(logits.device))
            else:
                group_probs = probs_option[mask]
                confidence_options[option].append(group_probs)                        
    return confidence_options