import torch
import torch.nn as nn
import torch.nn.functional as F

def safe_mean(tensor, default=0.0):
   if tensor.numel() == 0:
       return torch.tensor(default).to(tensor.device)
   return tensor.mean().nan_to_num(default)

def cwECE(logits, label, upper_boundary, tokenizer, extracted=False):
   logits = logits.detach().to('cpu')
   upper_boundary = upper_boundary.to('cpu')
   
   if logits.size(dim=0) == 0:
       return torch.zeros(1, requires_grad=True)
       
   if not extracted:
       label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
       label_logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
   else:
       label_logits = logits
       
   probs = F.softmax(label_logits, dim=-1).to('cpu')
   label = label.squeeze().to('cpu')
   label_matrix = torch.zeros_like(probs, dtype=torch.bool).to('cpu')
   
   # One hot
   for i in range(probs.size(dim=0)):
       label_matrix[i][label[i].item()] = True
   label_matrix = label_matrix.float()
   
   gt_count_list = torch.zeros((4, 10)).to('cpu')
   total_count_list = torch.zeros((4, 10)).to('cpu') 
   sum_conf_list = torch.zeros((4,10)).to('cpu')
   
   # Binning
   indices = torch.bucketize(probs, upper_boundary, right=True).squeeze().to('cpu')
   proportion_option = {'A':[], 'B':[], 'C':[], 'D':[]}
   
   for k in range(1, len(upper_boundary)+1):
       for idx, option in enumerate(proportion_option):
           indices_option = indices[:, idx]
           mask_option = (indices_option == k)
           label_matrix_option = label_matrix[:, idx][mask_option]
           
           if mask_option.sum() == 0:
               gt_count = torch.tensor(0.0)
               total_count = 0
               sum_conf_option = torch.tensor(0.0)
               mean_val = torch.tensor(0.0)
           else:
               sum_conf_option = probs[:, idx][mask_option].sum()
               gt_count = label_matrix_option.sum()
               total_count = label_matrix_option.size(0)
               mean_val = safe_mean(label_matrix_option, 0.0)
           
           gt_count_list[idx, k-1] += gt_count.nan_to_num(0.0)
           total_count_list[idx, k-1] += total_count
           sum_conf_list[idx, k-1] += sum_conf_option.nan_to_num(0.0)
           proportion_option[option].append(mean_val)
   
   for option in proportion_option.keys():
       stacked = torch.stack([x.nan_to_num(0.0) for x in proportion_option[option]])
       proportion_option[option] = stacked.nan_to_num(0.0)
   
   return (proportion_option,
           gt_count_list.nan_to_num(0.0),
           total_count_list.nan_to_num(0.0),
           sum_conf_list.nan_to_num(0.0))