import re
import os
import torch
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
)
from peft import AutoPeftModelForCausalLM
from preprocess import *
from accuracy_bin_all_sample_max_inf import *
from datasets import load_dataset
from group_dividing import get_adaptive_bins
from ploter import *
from accuracy_bin_all_sample_max_temp import ModelWithTemperature, calibrate_temperature, accuracy_all_sample_with_temperature
from util_llama3_tulu import get_model
from reward_max_conf import *
from confidence_max_nb import *
def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune LLaMA2 for Calibration with Temperature Scaling"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size for training and evaluation"
    )
    parser.add_argument("--alpha",
                    type=float,
                    default=0.9,
                    help="Weight of calibration loss"
                    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=10,
        help="Number of confidence bins"
    )
    
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Validation split ratio for temperature calibration"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=50,
        help="Maximum iterations for temperature optimization"
    )
    
    return parser.parse_args()

def prepare_dataloader(dataset, tokenizer, batch_size, val_split, max_seq_length=1024):
    def tokenize_function(examples):
        return tokenizer(
            examples["messages"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_seq_length
        )

    # Tokenize dataset
    # dataset = arc2_dataset
    dataset = load_dataset("#######################")
    dataset = dataset.map(format_chat_template)
    dataset = dataset.map(formatting_prompts_tokenize_inference)
    tokenized_dataset = dataset['test_1'].map(tokenize_function)
    # tokenized_dataset = dataset.map(tokenize_function).select(range(2000))
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'choices', 'messages'])
    
    # Split dataset into train (for temperature calibration) and test
    train_size = int(len(tokenized_dataset) * val_split)
    test_size = len(tokenized_dataset) - train_size
    train_dataset, test_dataset = random_split(tokenized_dataset, [train_size, test_size])
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=8,
        pin_memory=True
    )
    
    return train_dataloader, test_dataloader

def evaluate_model(model, dataloader, tokenizer, upper_boundary, device, model_name="Base Model"):
    with torch.no_grad():
        accuracy_bin, TP, Total, confidence_bins = accuracy_all_sample(
            dataloader, model, tokenizer, upper_boundary, device
        )
        accuracy_bin = accuracy_bin.nan_to_num()
        accuracy_bin_cpu = accuracy_bin.to('cpu')
        print(f"The overall group accuracy for {model_name} is {accuracy_bin_cpu}")
        histogram_plotter_single(TP, Total, confidence_bins)
        del accuracy_bin
        torch.cuda.empty_cache()
        
        return TP, Total, confidence_bins

def main():
    args = parse_args()
    print("Model loading start")
    
    # Setup device and paths
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = "####################"
    os.chdir(path)
    
    # Load model and tokenizer
    model, model_path = get_model()
    model.config.pretraining_tp = 1
    model = model.to(device)
    

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    
    
    # Load and prepare dataset
    dataset = load_dataset("aaronwzl/mcqa_calibration_dataset")
    # dataset = arc2_dataset
    upper_boundary = get_adaptive_bins(args.num_bins).to(device)
    
    # Prepare dataloaders
    train_dataloader, test_dataloader = prepare_dataloader(
        dataset, tokenizer, args.batch_size, args.val_split
    )
    
    # Evaluate base model
    print("\nEvaluating base model...")
    base_TP, base_Total, base_conf = evaluate_model(
        model, test_dataloader, tokenizer, upper_boundary, device, "Base Model"
    )
    

    # Temperature scaling
    print("\nCalibrating temperature...")
    model_temp = calibrate_temperature(
        model, train_dataloader, tokenizer, device, max_iter=args.max_iter
    )
    print(f"Optimal temperature: {model_temp.temperature.item()}")
    
    print("\nEvaluating calibrated model...")
  
    
    # Evaluate temperature-scaled model
    print("\nEvaluating temperature-scaled model...")
    temp_accuracy_bin, temp_TP, temp_Total, temp_conf = accuracy_all_sample_with_temperature(
        test_dataloader, model_temp, tokenizer, upper_boundary, device
    )
    temp_accuracy_bin = temp_accuracy_bin.nan_to_num()
    print(f"The overall group accuracy for temperature-scaled model is {temp_accuracy_bin.cpu()}")

    # Compare ECE before and after temperature scaling
    from ece_evaluator import ece_evaluator
    base_ece = ece_evaluator(base_TP.cpu(), base_Total.cpu(), base_conf.cpu())
    temp_ece = ece_evaluator(temp_TP.cpu(), temp_Total.cpu(), temp_conf.cpu())

    print("\nBase model ECE:", base_ece)
    # print("\nCalibrated model ECE:", cali_ece)
    print("Temperature-scaled model ECE:", temp_ece)
    # print("ECE Improvement:", base_ece - temp_ece)

if __name__ == "__main__":
    main()