import argparse
import re
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
   AutoModelForCausalLM,
   AutoTokenizer,
   AutoConfig,
   AdamW,
   BitsAndBytesConfig,
   DataCollatorForLanguageModeling,
   pipeline,
   logging,
   get_scheduler,
   LlamaTokenizer
)
from peft import LoraConfig, peft_model
import bitsandbytes as bnb
import accelerate
from ploter import *
from datasets import load_dataset, load_from_disk
from group_dividing import get_adaptive_bins
from ece_evaluator import ece_evaluator
from cwece_evaluator import cwECE
from torch.nn.parallel import DataParallel as DP
from tqdm.auto import tqdm
from reward_max_conf import group_accuracies
from peft import AutoPeftModelForCausalLM, PeftModel
from accuracy_bin_all_sample_max_inf import *
from preprocess import *
from util_olmo import *
# from preprocess_arc2 import *
# from perplexity import *
def parse_args():
   parser = argparse.ArgumentParser(
       description="Finetune LLaMA2 for Calibration"
       )
  
   parser.add_argument("--batch_size",
                       type=int,
                       default=3,
                       help="Batch size for training and evaluation"
                       )
  
   parser.add_argument("--num_bins",
                       type=int,
                       default=10,
                       help="Number of confidence bins")

   parser.add_argument("--alpha",
                       type=float,
                       default=0.9,
                       help="Weight of calibration loss"
                       )
   return parser.parse_args()


def main():
   args = parse_args()
   print("model loading start")
   #writer = SummaryWriter("/cbica/home/wangzha/LLM_Calibration/runs")
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   path = "####################"
   os.chdir(path)
   #####################Model & Tokenizer Setup####################################
   model_path = f"##########################"
   model = AutoPeftModelForCausalLM.from_pretrained(model_path,
                                                    torch_dtype=torch.float16,
                                                    device_map='auto')
   # model, model_path = get_model()
   model.config.pretraining_tp = 1
   model = model.to(device)
   tokenizer = AutoTokenizer.from_pretrained(model_path)
   tokenizer.pad_token = tokenizer.eos_token
   tokenizer.padding_side = 'left'
   tokenizer.truncation_side = 'left'
   ###################Dataset Processing########################################
   dataset = load_dataset("#############")
   dataset = dataset['test_1']
   # dataset = arc2_dataset
   print(dataset[0])
   upper_boundary = get_adaptive_bins(args.num_bins).to(device)
   max_seq_length = 1024
   ###################Dataloader Setup########################################
   def tokenize_function(examples):
      return tokenizer(examples["messages"], padding="max_length", truncation=True, max_length=max_seq_length)
   dataset = dataset.map(format_chat_template)
   dataset = dataset.map(formatting_prompts_tokenize_inference)
   # tokenized_dataset = dataset['test_1'].map(tokenize_function)
   tokenized_dataset = dataset.map(tokenize_function)
   # tokenized_dataset = tokenized_dataset.select(range(2000))
   tokenized_dataset = tokenized_dataset.remove_columns(['question', 'choices', 'messages'])
   data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
   eval_dataloader = DataLoader(tokenized_dataset, 
                              batch_size=args.batch_size,
                              shuffle=False,
                              collate_fn=data_collator,
                              num_workers=8,
                              pin_memory=True
                              # sampler=DistributedSampler(tokenized_dataset)
                              )
   
   
   with torch.no_grad():
      accuracy_bin, TP, Total, confidence_bins = accuracy_all_sample(eval_dataloader, model, tokenizer, upper_boundary, device)
      accuracy_bin = accuracy_bin.nan_to_num()
      accuracy_bin_cpu = accuracy_bin.to('cpu')
      print(f"The overall group accuracy is {accuracy_bin_cpu} for pure LLaMA2")
      histogram_plotter_single(TP, Total, confidence_bins)
      del accuracy_bin
      torch.cuda.empty_cache()
   # prediction = pd.DataFrame(prediction).to_csv("prediction.csv")
   # golden_label = pd.DataFrame(golden_label).to_csv("golden_label.csv")
   from ece_evaluator import ece_evaluator
   ece = ece_evaluator(TP.cpu(), Total.cpu(), confidence_bins.cpu())
   print(ece)
if __name__ == "__main__":
   main()


