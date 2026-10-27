import torch
import torch.nn as nn
from torch import Tensor
from ps2s.conf import MAX_ATOM_NUM, DECODER_HIDDEN_DIM, DECODER_HIDDEN_LAYERS, ATOM_FEATURES
from ps2s.utils.utils import dct_matrix as DCT
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Config
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer

class Decoder(nn.Module):

    def __init__(self, model_path: str='<PROJECT_ROOT>/ps2s-anonymous/PolymerGenerationPretrainedModelTrain/finetuned_polymer_model'):
        super().__init__()
        config = T5Config.from_pretrained(model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path, config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def properties_to_prompts(self, properties):
        if isinstance(properties, torch.Tensor):
            properties = properties.detach().cpu().numpy()
        prompts = []
        for p in properties:
            prompt = 'Generate polymer with properties: '
            prompt += ', '.join([f'P{i + 1}={v:.4f}' for i, v in enumerate(p)])
            prompts.append(prompt)
        return prompts

    def forward(self, properties, labels=None, return_smiles=False):
        prompts = self.properties_to_prompts(properties)
        enc = self.tokenizer(prompts, padding=True, truncation=True, return_tensors='pt')
        input_ids = enc['input_ids'].to(self.model.device)
        attention_mask = enc['attention_mask'].to(self.model.device)
        label_ids = None
        decoder_input_ids = None
        if labels is not None:
            if isinstance(labels, tuple):
                labels = list(labels)
            label_enc = self.tokenizer(labels, padding=True, truncation=True, return_tensors='pt')
            label_ids = label_enc['input_ids'].to(self.model.device)
            label_ids[label_ids == self.tokenizer.pad_token_id] = -100
            decoder_input_ids = self.model._shift_right(label_ids)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=label_ids)
        gen_smiles = None
        if return_smiles:
            with torch.no_grad():
                generated_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=200, do_sample=True, top_k=50, top_p=0.95)
                gen_smiles = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return (outputs, gen_smiles)
