import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from reward_max_conf import *
from confidence_max_nb import *
from accuracy_bin_all_sample_max import *
from trl import SFTTrainer
from not_max_target_confidence_new import *
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from trl import SFTTrainer
import torch.distributed as dist
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from torch.utils.data import DataLoader
from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling

class CustomCalibrationTrainer(SFTTrainer):
    def __init__(self, *args, num_bins=10, alpha=0.5, total_count=None, accuracy_all=None, device='cuda', train_dataloader=None, tokenizer, upper_boundary, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.num_bins = num_bins
        self.alpha = alpha
        self.upper_boundary = torch.linspace(0, 1, num_bins+1).to(device)
        self.accuracy = accuracy_all.to(device) if torch.is_tensor(accuracy_all) else torch.tensor(accuracy_all).to(device) if accuracy_all is not None else None
        self.total_count = total_count.to(device) if torch.is_tensor(total_count) else torch.tensor(total_count).to(device) if total_count is not None else None
        self.step = 0
        self.train_dataloader = train_dataloader
        self.all_confidences = []
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.tokenizer = tokenizer
        self.upper_boundary = upper_boundary
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits
        
        label_indices = {label: self.tokenizer.convert_tokens_to_ids(label) for label in ['A', 'B', 'C', 'D']}
        label_logits = logits[:, -2, [label_indices[label] for label in ['A', 'B', 'C', 'D']]]
        prob = F.softmax(label_logits, dim=-1)
        
        # Collect confidences
        self.collect_confidences(prob)
        
        sft_loss = super().compute_loss(model, inputs, return_outputs=False)
        calibration_loss = self.compute_calibration_loss(logits)
        total_loss = (1 - self.alpha) * sft_loss + self.alpha * calibration_loss *(1e-3)
        
        if self.step % 100 == 0 and dist.get_rank() == 0:
            print(f"\nStep {self.step}")
            print(f"SFT Loss: {sft_loss.detach().item():.8f}")
            print(f"Calibration Loss: {calibration_loss.item():.8f}")
            print(f"Total Loss: {total_loss.item():.8f}")
            print(f"Current confidence collection size: {len(self.all_confidences)}")
        
        self.step += 1
        return (total_loss, outputs) if return_outputs else total_loss
    
    def compute_calibration_loss(self, logits):
        if self.accuracy is None or self.total_count is None:
            return torch.tensor(0.0, device=self.device)
            
        loss = torch.tensor(0.0, device=self.device)
        max_confidence, not_max_confidence = group_confidences(logits, self.tokenizer, self.upper_boundary)
        
        for k in range(len(self.upper_boundary)-1):
            accuracy = self.accuracy[k]
            total_count_bin = self.total_count[k]
            
            group_max_confidence = max_confidence[k].to(self.device)
            group_not_max_confidence = not_max_confidence[k].to(self.device)
            
            if group_not_max_confidence.size() == torch.Size([]):
                continue
                
            group_not_max_target = not_max_target_confidence(accuracy, group_max_confidence, group_not_max_confidence, self.device)
            
            loss_k = (torch.square(group_max_confidence - accuracy).to(torch.float64).sum() + 
                     torch.square(group_not_max_confidence - group_not_max_target).to(torch.float64).sum())
            loss_k = loss_k/total_count_bin
            loss = loss + loss_k
            
        return loss
    
    def collect_confidences(self, prob):
        """Collect confidence scores from all processes"""
        if dist.is_initialized():
            gathered_probs = [torch.zeros_like(prob) for _ in range(self.world_size)]
            dist.all_gather(gathered_probs, prob)
            
            if dist.get_rank() == 0:  # Only collect on main process
                for p in gathered_probs:
                    probs_cpu = p.detach().cpu().numpy()
                    self.all_confidences.extend(probs_cpu)
        else:
            probs_cpu = prob.detach().cpu().numpy()
            self.all_confidences.extend(probs_cpu)
    
    def analyze_confidences(self):
        """Analyze collected confidence scores"""
        if not self.all_confidences:
            print("Warning: No confidence scores collected.")
            return None
        
        try:
            confidences = np.array(self.all_confidences)
            print(f"Analyzing {len(self.all_confidences)} confidence scores")
            print(f"Shape of confidence array: {confidences.shape}")
            
            sorted_confidences = np.sort(confidences, axis=1)[:, ::-1]
            
            avg_max = np.mean(sorted_confidences[:, 0])
            avg_second = np.mean(sorted_confidences[:, 1])
            avg_third = np.mean(sorted_confidences[:, 2])
            avg_fourth = np.mean(sorted_confidences[:, 3])
            
            return {
                'max_conf': sorted_confidences[:, 0],
                'second_conf': sorted_confidences[:, 1],
                'third_conf': sorted_confidences[:, 2],
                'fourth_conf': sorted_confidences[:, 3],
                'averages': {
                    'max': avg_max,
                    'second': avg_second,
                    'third': avg_third,
                    'fourth': avg_fourth
                }
            }
        except Exception as e:
            print(f"Error in analyze_confidences: {str(e)}")
            return None
    
    def plot_confidences(self, epoch):
        """Plot confidence analysis results for the current epoch"""
        if not dist.is_initialized() or dist.get_rank() == 0:
            results = self.analyze_confidences()
            if not results:
                return
            
            try:
                plt.figure(figsize=(12, 6))
                
                positions = ['Max', '2nd Max', '3rd Max', '4th Max']
                confidence_arrays = [
                    results['max_conf'],
                    results['second_conf'],
                    results['third_conf'],
                    results['fourth_conf']
                ]
                
                plt.boxplot(confidence_arrays, labels=positions)
                
                averages = [
                    results['averages']['max'],
                    results['averages']['second'],
                    results['averages']['third'],
                    results['averages']['fourth']
                ]
                plt.plot(range(1, 5), averages, 'r*', label='Mean', markersize=10)
                
                plt.xlabel('Confidence Rank')
                plt.ylabel('Confidence Value')
                plt.title(f'Distribution of Confidence Scores (Epoch {epoch})')
                plt.grid(True)
                plt.legend()
                
                plt.savefig(f'confidence_distribution_epoch_{epoch}.png')
                plt.close()
            except Exception as e:
                print(f"Error in plot_confidences: {str(e)}")

            
    def save_confidences(self, filename='confidence_scores.npy'):
        """Save confidence scores to file"""
        if not dist.is_initialized() or dist.get_rank() == 0:
            if self.all_confidences:
                try:
                    np.save(filename, np.array(self.all_confidences))
                    print(f"Saved confidence scores to {filename}")
                except Exception as e:
                    print(f"Error saving confidence scores: {str(e)}")
            else:
                print("No confidence scores to save")
    
    def train(self, resume_from_checkpoint=None, **kwargs):
        """Override train method to add epoch-specific processing"""
        num_epochs = int(self.args.num_train_epochs)
        
        for epoch in range(num_epochs):
            print(f"\n{'='*50}")
            print(f"Starting Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*50}")
            
            self.all_confidences = []

            super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"\n{'='*50}")
                print(f"End of Epoch {epoch+1}/{num_epochs}")
                print(f"{'='*50}")
                
                self.plot_confidences(epoch=epoch+1)
                self.save_confidences(f'confidence_scores_epoch_{epoch+1}.npy')
                print(f"{'='*50}\n")