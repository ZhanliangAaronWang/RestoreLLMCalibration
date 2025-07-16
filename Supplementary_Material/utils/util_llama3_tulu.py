import torch
from transformers import (AutoProcessor,
                         AutoModelForCausalLM
                        )
from transformers import BitsAndBytesConfig
from huggingface_hub import login
from peft import (
   LoraConfig,
   PeftModel,
   prepare_model_for_kbit_training,
   get_peft_model
)
import bitsandbytes as bnb
import loralib as lora
from accelerate import PartialState
# device_string = PartialState().process_index
def find_all_linear_names(model):
   cls = bnb.nn.Linear4bit
   lora_module_names = set()
   for name, module in model.named_modules():
       if isinstance(module, cls):
           names = name.split('.')
           lora_module_names.add(names[0] if len(names) == 1 else names[-1])
   if 'lm_head' in lora_module_names:  # needed for 16 bit
       lora_module_names.remove('lm_head')
   return list(lora_module_names)



def getLoraModel(model):
  # Define LoRA Config
  peft_config = LoraConfig(
      task_type="CAUSAL_LM",
      inference_mode=False,
      r=128,
      lora_alpha=64,
      lora_dropout=0.05,
      bias='none',
   #    target_modules=['q_proj', 'v_proj']
      target_modules='all-linear'
  )
  # add LoRA adaptor
  model.enable_input_require_grads()
  model = get_peft_model(model, peft_config)
  lora.mark_only_lora_as_trainable(model)
  model.print_trainable_parameters()
  return model


def get_model():
    login(token="***************")
    model_path = 'allenai/Llama-3.1-Tulu-3-8B-DPO'
    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    )
    model = getLoraModel(model)
    return model, model_path
    
