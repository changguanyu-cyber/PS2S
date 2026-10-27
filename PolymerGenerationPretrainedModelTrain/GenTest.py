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
path = '<PROJECT_ROOT>/ps2s-anonymous/PolymerGenerationPretrainedModelTrain/PolymerGenerationPretrainedModel'
tokenizer = AutoTokenizer.from_pretrained(path, pad_token='[PAD]', padding_side='right')
configuration = T5Config.from_pretrained(path, output_hidden_states=False)
model = T5ForConditionalGeneration.from_pretrained(path, config=configuration)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()
data = np.load('<PROJECT_ROOT>/ps2s/data/PI1M_v2_pretrain.npz', allow_pickle=True)
model_path = '<PROJECT_ROOT>/ps2s-anonymous/ckpt'
smiless = [str(x) for x in data['smiles']]
properties = [str(x) for x in data['properties'][2]]
print(properties)
print(smiless[2])
prompt = '500,15,3,7,11,0,0,0,0,4,0,0,0,0,0'
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)
sample_outputs = model.generate(generated, do_sample=True, top_k=3, max_length=300, top_p=0.95, num_return_sequences=5)
for i, sample_output in enumerate(sample_outputs):
    print('{}: {}\n\n'.format(i + 1, tokenizer.decode(sample_output, skip_special_tokens=True)))
