import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from reward_max_conf import *
from confidence_max_nb import *
from cwece_evaluator import cwECE
import gc
from ploter import *
class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, **kwargs):
        outputs = self.model(**kwargs)
        return outputs

    def scale_logits(self, logits):
        return logits / self.temperature

def monitor_memory(message=""):
    print(f"\nMemory Usage {message}:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.memory_allocated(i)/1e9:.2f}GB allocated, "
              f"{torch.cuda.memory_reserved(i)/1e9:.2f}GB reserved")
def calibrate_temperature(model, valid_loader, tokenizer, device, max_iter=50):
    torch.cuda.empty_cache()
    model_temp = ModelWithTemperature(model)
    model_temp.to(device)
    
    optimizer = optim.LBFGS([model_temp.temperature], lr=0.01, max_iter=max_iter)
    
    label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}

    def eval_loss():
        optimizer.zero_grad()
        total_loss = 0
        n_samples = 0
        
        for batch in valid_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "answer"}
            labels = batch["answer"].to(device)  
            
            with torch.no_grad():
                outputs = model_temp(**inputs)
                logits = outputs.logits
                label_logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
                scaled_logits = model_temp.scale_logits(label_logits)
                loss = F.cross_entropy(scaled_logits, labels)
            
            total_loss += loss.item() * len(labels)
            n_samples += len(labels)
            
            del outputs, logits, label_logits, scaled_logits
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / n_samples
        loss = torch.tensor(avg_loss, requires_grad=True, device=device)
        loss.backward()
        return loss

    optimizer.step(eval_loss)
    return model_temp

def accuracy_all_sample_with_temperature(dataloader, model_temp, tokenizer, upper_boundary, device):
    torch.cuda.empty_cache()
    model_temp.eval()
    
    upper_boundary = upper_boundary.cpu()
    correct_count_bin_all = torch.zeros(upper_boundary.size(dim=0))
    count_bin_all = torch.zeros(upper_boundary.size(dim=0))
    confidence_mean_all = torch.zeros(upper_boundary.size(dim=0))
    
    ground_truth_prop_all = torch.zeros((4, 10))
    total_prop_all = torch.zeros((4, 10))
    confidence_mean_prop_all = torch.zeros((4,10))
    
    label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
    original_max_confidence = []
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "answer"}
            labels = batch["answer"]  
            
            with torch.no_grad():
                outputs = model_temp(**inputs)
                logits = outputs.logits
                print("The -1 place", tokenizer.decode(torch.argmax(logits, dim=-1)[0][-1]),
                      "The -2 place", tokenizer.decode(torch.argmax(logits, dim=-1)[0][-2]),
                      "The -3 place", tokenizer.decode(torch.argmax(logits, dim=-1)[0][-3]),
                      "The -4 place", tokenizer.decode(torch.argmax(logits, dim=-1)[0][-4]),
                      "The -5 place", tokenizer.decode(torch.argmax(logits, dim=-1)[0][-5]))
            
                label_logits = logits[:, -1, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
                scaled_logits = model_temp.scale_logits(label_logits).cpu()
                
                del outputs, logits, label_logits
                torch.cuda.empty_cache()
                

                _, count_bin, correct_count_bin, group_confidence_mean = group_accuracies(
                    scaled_logits, 
                    labels,  
                    tokenizer,
                    upper_boundary, 
                    extracted=True,
                    inference=True
                )
                
                _, ground_truth_prop, total_prop, confidence_mean_prop = cwECE(
                    scaled_logits,
                    labels,
                    upper_boundary,
                    tokenizer,
                    extracted=True
                )
                
                group_confidence_mean = torch.tensor(group_confidence_mean)
                group_confidence_mean = torch.multiply(group_confidence_mean, count_bin)
                
                count_bin_all += count_bin
                correct_count_bin_all += correct_count_bin
                confidence_mean_all += group_confidence_mean
                
                ground_truth_prop_all += ground_truth_prop
                total_prop_all += total_prop
                confidence_mean_prop_all += confidence_mean_prop
                
                original_max_confidence.append(scaled_logits)
                
        except RuntimeError as e:
            print(f"Error in batch processing: {e}")
            torch.cuda.empty_cache()
            continue
            
        if batch_idx % 10 == 0:
            gc.collect()
    
    try:
        confidence_mean_all = torch.div(confidence_mean_all, count_bin_all)
        accuracy_bin = torch.div(correct_count_bin_all, count_bin_all)
        
        original_max_confidence = torch.cat(original_max_confidence)
        original_max_confidence = F.softmax(original_max_confidence, dim=-1)
        
        proportion_bin = torch.div(ground_truth_prop_all, total_prop_all)
        confidence_mean_prop_all = torch.div(confidence_mean_prop_all, total_prop_all).nan_to_num(0.0)
        cwece = (torch.abs(proportion_bin - confidence_mean_all) * 
                total_prop_all/count_bin_all.sum()).nan_to_num().sum()/4
        
        print(f"TP counts for all samples over bins: {correct_count_bin_all}; "
              f"Counts for all samples over bins: {count_bin_all}; "
              f"Average confidence for all samples over bins:{confidence_mean_all}")
        
        print(f"Ground truth proportion: {ground_truth_prop_all}",
              f"Total count proportion: {total_prop_all}",
              f"Average confidence for proportion: {confidence_mean_prop_all}")
        print("classwise ECE for Temperature Scaling is: ", cwece)
        
        return accuracy_bin, correct_count_bin_all, count_bin_all, confidence_mean_all
        
    except Exception as e:
        print(f"Error in final computations: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()