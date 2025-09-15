import torch
from reward_max_conf import group_accuracies
import torch.nn.functional as F
from cwece_evaluator import cwECE
import gc
import pandas as pd
from ploter import *
def accuracy_all_sample(dataloader, model, tokenizer, upper_boundary, device):
    upper_boundary = upper_boundary.to('cpu')
    correct_count_bin_all = torch.zeros(upper_boundary.size(dim=0))
    count_bin_all = torch.zeros(upper_boundary.size(dim=0))
    confidence_mean_all = torch.zeros(upper_boundary.size(dim=0))
    
    ground_truth_prop_all = torch.zeros((4, 10))
    total_prop_all = torch.zeros((4, 10))
    confidence_mean_prop_all = torch.zeros((4, 10))
    prediction = []
    golden_label = []
    label_indices = {label: tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
    original_max_confidence = []
    
    model.eval()
    
    for batch in dataloader:
        try:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "answer"}
            labels = batch["answer"] 
            golden_label.append(labels)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predict = tokenizer.decode(torch.argmax(logits, dim=-1)[0][-1])
                prediction.append(predict)
                del outputs
                torch.cuda.empty_cache()
                
                _, count_bin, correct_count_bin, group_confidence_mean = group_accuracies(
                    logits.cpu(), 
                    labels,
                    tokenizer,
                    upper_boundary,
                    extracted=False,
                    inference=False
                )
                
                _, ground_truth_prop, total_prop, confidence_mean_prop = cwECE(
                    logits.cpu(), 
                    labels,
                    upper_boundary,
                    tokenizer,
                    extracted=False
                )

                group_confidence_mean = torch.tensor(group_confidence_mean)
                group_confidence_mean = torch.multiply(group_confidence_mean, count_bin)
                
                count_bin_all += count_bin
                correct_count_bin_all += correct_count_bin
                confidence_mean_all += group_confidence_mean
                
                ground_truth_prop_all += ground_truth_prop
                total_prop_all += total_prop
                confidence_mean_prop_all += confidence_mean_prop

                
        except RuntimeError as e:
            print(f"Error in batch processing: {e}")
            torch.cuda.empty_cache()
            continue
        
        if len(original_max_confidence) % 10 == 0:
            gc.collect()
    
    try:
        confidence_mean_all = torch.div(confidence_mean_all, count_bin_all)
        accuracy_bin = torch.div(correct_count_bin_all, count_bin_all)
        
        if original_max_confidence:
            batch_size = 32
            max_conf_batches = []
            for i in range(0, len(original_max_confidence), batch_size):
                batch = original_max_confidence[i:i + batch_size]
                batch_tensor = torch.cat(batch)
                max_conf_batches.append(F.softmax(batch_tensor, dim=-1))
            original_max_confidence = torch.cat(max_conf_batches)
        
        proportion_bin = torch.div(ground_truth_prop_all, total_prop_all)
        confidence_mean_bin = torch.div(confidence_mean_prop_all, total_prop_all).nan_to_num(0.0)
        cwece = (torch.abs(proportion_bin - confidence_mean_bin) * 
                total_prop_all/count_bin_all.sum()).nan_to_num().sum()/4
        
        print("\nResults:")
        print(f"Ground truth proportion: {ground_truth_prop_all}")
        print(f"Total count proportion: {total_prop_all}")
        print(f"Average confidence for proportion: {confidence_mean_bin}")
        print(f"classwise ECE for Temperature Scaling is: {cwece}")
        print(f"TP counts: {correct_count_bin_all}")
        print(f"Total counts: {count_bin_all}")
        print(f"Average confidence: {confidence_mean_all}")
        
        histogram_plotter_cw(ground_truth_prop_all, total_prop_all, confidence_mean_bin)
        return accuracy_bin, correct_count_bin_all, count_bin_all, confidence_mean_all
        
    except Exception as e:
        print(f"Error in final computations: {e}")
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()