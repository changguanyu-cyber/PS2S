import os
import time
import datetime
import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import warnings
warnings.filterwarnings('ignore')
import nltk

class GPT2Dataset(Dataset):

    def __init__(self, propmt_list, answer_list, tokenizer, max_length_propmt=1024, max_length_answer=10):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.answer_ids = []
        tokenizer.padding_side = 'right'
        encodings_list1 = [tokenizer('<s>' + txt, truncation=True, max_length=max_length_propmt, padding='max_length') for txt in propmt_list]
        self.input_ids = [torch.tensor(encodings_dict['input_ids']) for encodings_dict in encodings_list1]
        self.attn_masks = [torch.tensor(encodings_dict['attention_mask']) for encodings_dict in encodings_list1]
        encodings_list2 = [tokenizer('<s>' + str(ans), truncation=True, max_length=max_length_answer, padding='max_length') for ans in answer_list]
        self.answer_ids = [torch.tensor(encodings_dict['input_ids']) for encodings_dict in encodings_list2]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attn_masks[idx], self.answer_ids[idx])
path = '<PROJECT_ROOT>/ps2s-anonymous/PolymerGenerationPretrainedModelTrain/finetuned_polymer_model'
tokenizer = AutoTokenizer.from_pretrained(path, pad_token='[PAD]', padding_side='right')
configuration = T5Config.from_pretrained(path, output_hidden_states=False)
model = T5ForConditionalGeneration.from_pretrained(path, config=configuration)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
data = np.load('<PROJECT_ROOT>/ps2s/data/PI1M_v2_pretrain.npz', allow_pickle=True)
model_path = '<PROJECT_ROOT>/ps2s-anonymous/ckpt'
import torch
import numpy as np
from rdkit import Chem
prompt = '300,37,6,8,26,1,1,0,0,0,0,0,0,0,0'
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
for k in [3, 5, 10]:
    print(f'\n======================')
    print(f'Top-k = {k}')
    print(f'======================')
    outputs = model.generate(generated, do_sample=True, top_k=k, top_p=0.95, max_length=300, num_return_sequences=20, return_dict_in_generate=True, output_scores=True)
    sequences = outputs.sequences
    scores = outputs.scores
    smiles_list = tokenizer.batch_decode(sequences, skip_special_tokens=True)
    valid_count = 0
    all_ranks = []
    critical_tokens = []
    for i in range(len(smiles_list)):
        smi = smiles_list[i]
        mol = Chem.MolFromSmiles(smi)
        is_valid = mol is not None
        if is_valid:
            valid_count += 1
        token_ranks = []
        for t, score in enumerate(scores):
            probs = torch.softmax(score, dim=-1)
            token_id = sequences[i, t + 1]
            sorted_probs, sorted_idx = torch.sort(probs[i], descending=True)
            rank = (sorted_idx == token_id).nonzero(as_tuple=True)[0].item() + 1
            token_ranks.append(rank)
            all_ranks.append(rank)
            if 5 < rank <= 10:
                critical_tokens.append(rank)
        if i < 3:
            print(f'\nSample {i + 1}:')
            print('SMILES:', smi[:120])
            print('Valid:', is_valid)
            print('First 10 ranks:', token_ranks[:10])
    total = len(all_ranks)
    top3 = sum([r <= 3 for r in all_ranks]) / total
    top5 = sum([r <= 5 for r in all_ranks]) / total
    top10 = sum([r <= 10 for r in all_ranks]) / total
    critical_ratio = len(critical_tokens) / total
    print('\n===== Statistics =====')
    print(f'Validity: {valid_count / len(smiles_list):.4f}')
    print(f'Top-3 coverage: {top3:.4f}')
    print(f'Top-5 coverage: {top5:.4f}')
    print(f'Top-10 coverage: {top10:.4f}')
    print(f'Critical tokens (5<rank<=10): {critical_ratio:.4f}')
print('\n======================')
print('Invalid SMILES Analysis')
print('======================')
outputs = model.generate(generated, do_sample=True, top_k=5, top_p=0.95, max_length=300, num_return_sequences=50, return_dict_in_generate=True, output_scores=True)
sequences = outputs.sequences
scores = outputs.scores
smiles_list = tokenizer.batch_decode(sequences, skip_special_tokens=True)
invalid_ranks = []
for i in range(len(smiles_list)):
    smi = smiles_list[i]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        for t, score in enumerate(scores):
            probs = torch.softmax(score, dim=-1)
            token_id = sequences[i, t + 1]
            sorted_probs, sorted_idx = torch.sort(probs[i], descending=True)
            rank = (sorted_idx == token_id).nonzero(as_tuple=True)[0].item() + 1
            invalid_ranks.append(rank)
if len(invalid_ranks) > 0:
    invalid_top5 = sum([r <= 5 for r in invalid_ranks]) / len(invalid_ranks)
    invalid_top10 = sum([r <= 10 for r in invalid_ranks]) / len(invalid_ranks)
    print(f'Invalid tokens Top-5 coverage: {invalid_top5:.4f}')
    print(f'Invalid tokens Top-10 coverage: {invalid_top10:.4f}')
