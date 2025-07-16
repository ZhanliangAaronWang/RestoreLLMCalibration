import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.pipelines.pt_utils import KeyDataset
import multiprocessing
from tqdm import tqdm
from set_seed import *
import subprocess
import numpy as np
from accelerate import PartialState
from util_olmo import *
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          DataCollatorForLanguageModeling
                         )
from accuracy_bin_all_sample_max import *
from reward_max_conf import *
from confidence_max_nb import *
from preprocess import *
from group_dividing import *
from not_max_target_confidence_new import *
from Custom_Trainer import *
import argparse
import os
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from trl import SFTConfig, DataCollatorForCompletionOnlyLM
from transformers import DataCollatorForLanguageModeling
from accuracy_bin_all_sample_max import accuracy_all_sample
from Custom_Trainer import CustomCalibrationTrainer
from preprocess import format_chat_template, formatting_prompts_func, formatting_prompts_tokenize
from set_seed import set_seed
from ece_evaluator import ece_evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune LLM for Calibration")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--model_path", type=str, default="/path/to/model")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    return parser.parse_args()

def set_ddp():
    if torch.cuda.is_available():
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
    
def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()

def main():
    # Initialize args and setup
    args = parse_args()
    set_seed(42)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup DDP
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    set_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Model & Tokenizer Setup
        model, model_path = get_model()
        model.config.pretraining_tp = 1
        model = model.to(device)
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'
        
        # Dataset Processing
        dataset = load_dataset("*************")
        upper_boundary = get_adaptive_bins(args.num_bins).to(device)
        max_seq_length = 2048
        
        # Apply preprocessing
        dataset = dataset.map(format_chat_template)
        print(f"Sample after format_chat_template: {dataset['calibration_train'][0]}")
        dataset = dataset.map(formatting_prompts_tokenize)
        
        # Setup dataloader
        def tokenize_function(examples):
            return tokenizer(
                examples["messages"],
                padding="max_length",
                truncation=True,
                max_length=max_seq_length
            )
        
        tokenized_dataset = dataset['calibration_train'].map(tokenize_function)
        tokenized_dataset = tokenized_dataset.remove_columns(['question', 'choices', 'messages'])
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        train_dataloader = DataLoader(
            tokenized_dataset, 
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=8,
            pin_memory=True
        )

        # Calculate initial accuracy
        print("Calculating initial accuracy...")
        with torch.no_grad():
            accuracy_bin, _, total_count, _ = accuracy_all_sample(
                train_dataloader,
                model,
                tokenizer,
                upper_boundary,
                device
            )
            accuracy_bin = accuracy_bin.nan_to_num()
            accuracy_bin_cpu = accuracy_bin.to('cpu')
            print(f"Initial group accuracy: {accuracy_bin_cpu}")
            del accuracy_bin
            torch.cuda.empty_cache()

        # Training Configuration
        response_template = " ### Answer:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        
        training_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=args.learning_rate,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=25,
            save_steps=500,
            save_total_limit=2,
            evaluation_strategy='no',
            do_eval=False,
            bf16=True,
            push_to_hub=False,
            local_rank=local_rank,
            ddp_find_unused_parameters=False,
            use_liger=True
        )

        # Initialize trainer
        trainer = CustomCalibrationTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset['calibration_train'],
            tokenizer=tokenizer,
            upper_boundary=upper_boundary,
            max_seq_length=max_seq_length,
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
            num_bins=args.num_bins,
            alpha=args.alpha,
            device=device,
            accuracy_all=accuracy_bin_cpu,
            total_count=total_count,
            train_dataloader=train_dataloader,
            formatting_func=formatting_prompts_func,
            data_collator=collator,
        )

        # Training
        print("Starting training...")
        trainer.model.train()
        trainer.train()
        
        # Save model and analyze results
        if local_rank == 0:
            # trainer.plot_confidences()
            confidence_results = trainer.analyze_confidences()
            
            if confidence_results:
                print("\nConfidence Analysis Results:")
                print(f"Max confidence average: {confidence_results['averages']['max']:.4f}")
                print(f"Second max average: {confidence_results['averages']['second']:.4f}")
                print(f"Third max average: {confidence_results['averages']['third']:.4f}")
                print(f"Fourth max average: {confidence_results['averages']['fourth']:.4f}")

            save_path = os.path.join(args.output_dir, f'llama2_alpha_{args.alpha}')
            trainer.model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            trainer.save_model()
            
            # Calculate final metrics
            print("\nCalculating final metrics...")
            with torch.no_grad():
                accuracy_bin, TP, total_count, conf = accuracy_all_sample(
                    train_dataloader,
                    model,
                    tokenizer,
                    upper_boundary,
                    device
                )
                accuracy_bin = accuracy_bin.nan_to_num()
                accuracy_bin_cpu = accuracy_bin.to('cpu')
                print(f"Final group accuracy: {accuracy_bin_cpu}")
                
                base_ece = ece_evaluator(TP.cpu(), total_count.cpu(), conf.cpu())
                print(f"Final ECE score: {base_ece}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise e
    
    finally:
        cleanup_ddp()

if __name__ == "__main__":
    main()