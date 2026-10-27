import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
model_path = '<PROJECT_ROOT>/ps2s-anonymous/PolymerGenerationPretrainedModelTrain/PolymerGenerationPretrainedModel'
csv_path = 'data.csv'
save_path = '<PROJECT_ROOT>/ps2s-anonymous/PolymerGenerationPretrainedModelTrain/finetuned_polymer_model_2'
batch_size = 8
epochs = 50
lr = 1e-05
max_input_len = 128
max_output_len = 256
train_ratio = 0.9
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
data = np.load('<PROJECT_ROOT>/ps2s/data/PI1M_v2_pretrain.npz', allow_pickle=True)
properties = [str(x) for x in data['properties'][:8000]]
smiles = [str(x) for x in data['smiles'][:8000]]

class PolymerDataset(Dataset):

    def __init__(self, inputs, targets, tokenizer):
        self.tokenizer = tokenizer
        self.model_inputs = tokenizer(inputs, max_length=max_input_len, padding='max_length', truncation=True, return_tensors='pt')
        labels = tokenizer(targets, max_length=max_output_len, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        labels[labels == tokenizer.pad_token_id] = -100
        self.labels = labels

    def __len__(self):
        return self.model_inputs['input_ids'].shape[0]

    def __getitem__(self, idx):
        return {'input_ids': self.model_inputs['input_ids'][idx], 'attention_mask': self.model_inputs['attention_mask'][idx], 'labels': self.labels[idx]}
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = T5Config.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
model.to(device)
dataset = PolymerDataset(properties, smiles, tokenizer)
train_size = int(len(dataset) * train_ratio)
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
optimizer = AdamW(model.parameters(), lr=lr)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    print(f'\n======== Epoch {epoch + 1} / {epochs} ========')
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)
    print(f'Training Loss: {avg_train_loss:.4f}')
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            total_val_loss += outputs.loss.item()
    avg_val_loss = total_val_loss / len(val_loader)
    print(f'Validation Loss: {avg_val_loss:.4f}')
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print('Training completed; model saved.')
