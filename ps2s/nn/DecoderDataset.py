import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class PolymerDataset(Dataset):

    def __init__(self, inputs, targets, tokenizer):
        self.tokenizer = tokenizer
        self.model_inputs = tokenizer(inputs, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        labels = tokenizer(targets, max_length=128, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100
        self.labels = labels

    def __len__(self):
        return self.model_inputs['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.model_inputs['input_ids'][idx], 'attention_mask': self.model_inputs['attention_mask'][idx], 'labels': self.labels[idx]}
