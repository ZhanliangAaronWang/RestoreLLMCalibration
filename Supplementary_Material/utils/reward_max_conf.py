import torch
import torch.nn.functional as F

def group_accuracies(logits, labels, tokenizer, upper_boundary, extracted=False, inference=False):
    """Calculates accuracy for each group."""
    logits = logits.detach() # frozen, to let the confidence move towards the fixed accuracy, accuracy<->conf?
    if logits.size(dim=0) == 0:
        return torch.tensor(0.0)
    if not extracted:
        if not inference:
            label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
            label_logits = logits[:, -2, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
        else:
            label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
            label_logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D']]] 
    else:
        label_logits = logits
    
    probs = F.softmax(label_logits, dim=-1)
    
    max_confidence_index = torch.argmax(probs, dim=-1)
    max_confidence = torch.gather(probs, dim=-1, index=max_confidence_index.unsqueeze(-1))
    # Create pseudo distribution matrix for confidence, and label
    new_confidence = torch.zeros_like(probs, dtype=torch.bool)
    new_label = torch.zeros_like(probs, dtype=torch.bool)

    indices = torch.bucketize(max_confidence, upper_boundary, right=True).squeeze()# include the left boundary but not right boundary point
    ##########################################################################################################################################
    # Prediction probability distribution matrix and label one hot distribution matrix
    for i in range(probs.size(dim=0)):
        new_confidence[i][max_confidence_index[i]] = True # This is the prediction for 4*m samples (True/False Pair)
        new_label[i][labels[i]] = True

    new_label_max_confidence = torch.gather(new_label, dim=-1, index=max_confidence_index.unsqueeze(-1))
    #############################################################################################################################################
    group_accuracy = []
    correct_count_bin = []
    count_bin = []
    group_max_confidence_mean = []
    for k in range(1, len(upper_boundary)+1):
        mask = (indices == k) # samples distributed into bin k
        if mask.sum().item() == 0:
            group_accuracy.append(torch.zeros(1, requires_grad=True)[0].to(logits.device))
            corrent_count_per_bin = 0
            count_per_bin = 0
            count_bin.append(count_per_bin)
            correct_count_bin.append(corrent_count_per_bin)
            group_max_confidence_mean.append(torch.zeros(1, requires_grad=True)[0].to(logits.device))
        else:
            max_confidence_mean = max_confidence[mask].float().mean().item()
            group_true_positive = new_label_max_confidence[mask].float() #False: 0, True: 1
            accuracy = group_true_positive.mean()
            correct_count_per_bin = group_true_positive.sum().item()
            count_per_bin = mask.sum().item()
            count_bin.append(count_per_bin)
            correct_count_bin.append(correct_count_per_bin)
            group_accuracy.append(accuracy)
            group_max_confidence_mean.append(max_confidence_mean)
    count_bin = torch.tensor(count_bin)
    correct_count_bin = torch.tensor(correct_count_bin)
    return group_accuracy, count_bin, correct_count_bin, group_max_confidence_mean