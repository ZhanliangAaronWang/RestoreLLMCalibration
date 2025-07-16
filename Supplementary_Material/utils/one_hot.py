import torch
import torch.nn.functional as F

def one_hot(logits, tokenizer, labels):
    label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
    logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
    logits1 = logits.detach()
    one_hot_matrix = torch.zeros_like(logits1, dtype=float)
    for i in range(logits1.size(0)):
        one_hot_matrix[i][labels[i].item()] = 1
    # one_hot_matrix.requires_grad_(True)
    return one_hot_matrix

