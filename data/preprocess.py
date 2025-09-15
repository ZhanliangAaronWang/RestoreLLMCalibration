from typing import Dict, Union, List
from transformers import PreTrainedTokenizer
from datasets import Dataset, load_dataset
import torch
import re
class DatasetFormatter:
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset: Dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset

    def format_dataset(self, example):
        # print(example['choices'])
        if len(example['choices']) == 4:
            prefixes = ['A. ', 'B. ', 'C. ', 'D. ']
            example['choices'] = [prefixes[i] + choice for i, choice in enumerate(example['choices'])]
        else:
            prefixes = ['A. ', 'B. ', 'C. ', 'D. ', 'E. ']
            example['choices'] = [prefixes[i] + choice for i, choice in enumerate(example['choices'])]
        return example

    def multiple_choice(self, inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
        PROMPT_FORMAT = 'Question:\n {query}\nOptions:{options}\nAnswer: '
        options = ''
        assert isinstance(inp['choices'], List)
        for option in inp['choices']:
            options += f'\n - {option}'
        query = inp['question']
        print(inp['choices'])
        if len(inp['choices']) == 4:
            label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        else: 
            label_mapping = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E'}
        print(type(inp['answer']), inp['answer'])
        assert isinstance(inp['answer'], int)
        return {
            'prompt': PROMPT_FORMAT.format(query=query, options=options),
            'response': label_mapping[inp['answer']],
        }
    
    def format(self, example):
        formatted_example = self.multiple_choice(self.format_dataset(example))
        return formatted_example
    
    def processed_dataset(self):
        formatted_dataset = self.dataset.map(self.format)
        return formatted_dataset
    def format_and_tokenize(self, example):
        formatted_example = self.multiple_choice(self.format_dataset(example))
        return self.tokenizer(text=formatted_example['prompt'], text_pair=formatted_example['response'])
    
    def tokenize_dataset(self):
        tokenized_dataset = self.dataset.map(self.format_and_tokenize)
        tokenized_dataset = tokenized_dataset.remove_columns(['question', 'choices'])
        return tokenized_dataset
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = f"### Question: {example['question'][i]}\n ### Answer: {example['response'][i]}"
        output_texts.append(text)
    return output_texts
import re
from typing import Dict, Union
import string

def format_chat_template(example: Dict[str, Union[str, int]]) -> Dict[str, list]:
    try:
        # print("Processing example:")
        # print(f"Question: {example['question']}")
        # print(f"Choices: {example['choices']}")
        # print(f"Answer: {example['answer']}")
        
        if isinstance(example['choices'], str):
            choices = re.findall(r"'([^']*)'", example['choices'])
        else:
            choices = example['choices']
        
        
        answer_idx = int(example['answer'])
        if answer_idx >= len(choices):
            # print(f"Warning: answer index {answer_idx} is out of range for {len(choices)} choices")
            answer_idx = answer_idx % len(choices)
        
        options = {i: letter for i, letter in enumerate(string.ascii_uppercase[:len(choices)])}
        # print(f"Options mapping: {options}")
        
        formatted_choices = '\n'.join([f"{options[i]}. {choice.strip()}" for i, choice in enumerate(choices)])
        
        formatted_question = f"""Question: {example['question']}

Options:
{formatted_choices}

Answer with only a single letter:"""

        correct_answer = options[answer_idx]

        return {
            'messages': [
                {
                    'role': "system",
                    'content': "You are an AI assistant that answers multiple choice questions. You must only respond with a single letter corresponding to your choice without any explanation or additional text."
                },
                {
                    'role': "user",
                    'content': formatted_question
                },
                {
                    'role': "assistant",
                    'content': correct_answer
                }
            ]
        }
    except Exception as e:
        print(f"Error processing example: {example}")
        print(f"Error details: {str(e)}")
        raise
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['question'])):
        text = example['messages'][i]
        output_texts.append(text)
    return output_texts
def formatting_prompts_tokenize(example):
    example['messages'] = f"### Question: {example['messages'][-2]['content']}\n ### Answer:{example['messages'][-1]['content']}"
    return example
def formatting_prompts_tokenize_inference(example):
    example['messages'] = f"### Question: {example['messages'][-2]['content']}\n ### Answer:"
    return example

def split_choices(example):
    choices_str = example['choices']
    example['choices'] = re.findall(r'\'(.*?)\'|\"(.*?)\"', choices_str)
    example['choices'] = [x[0] if x[0] else x[1] for x in example['choices']]
    return example