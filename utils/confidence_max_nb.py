import torch
import torch.nn.functional as F

def group_confidences(logits, tokenizer, upper_boundary, inference=False):
                                           
    if logits.size(dim=0) == 0:
        return torch.zeros(1, requires_grad=True)
    if not inference:
        label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
        label_logits = logits[:, -2, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
    else:
        label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
        label_logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
    probs = torch.exp(label_logits - torch.logsumexp(label_logits, dim=-1, keepdim=True))
    max_logits_idx = torch.argmax(label_logits, dim=-1, keepdim=True)
    max_probs = torch.gather(probs, dim=-1, index=max_logits_idx)
    not_max_confidence = torch.sort(probs, dim=-1, descending=True)[0][:,1:5]
    indices = torch.bucketize(max_probs, upper_boundary, right=True).squeeze()
    group_not_max_confidence = []
    group_max_confidence = []
    for k in range(1, len(upper_boundary)+1):
        mask = (indices == k)
        if mask.sum().item() == 0:
            group_not_max_confidence.append(torch.zeros(1, requires_grad=True)[0].to(logits.device))
            group_max_confidence.append(torch.zeros(1, requires_grad=True)[0].to(logits.device))
        else:
            group_max_confidence.append(max_probs[mask])
            group_not_max_confidence.append(not_max_confidence[mask])
            
    return group_max_confidence, group_not_max_confidence