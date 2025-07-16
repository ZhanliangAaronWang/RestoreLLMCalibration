import torch
def ece_evaluator(TP_count, Total_count, confidence_bins):
    accuracy = torch.div(TP_count, Total_count)
    weight = torch.div(Total_count, Total_count.sum())
    ece = torch.multiply(weight, (accuracy - confidence_bins).abs()).nan_to_num().sum()
    accuracy_batches = torch.multiply(weight, accuracy).nan_to_num().sum()
    return ece, accuracy_batches