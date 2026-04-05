import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import numpy as np

class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts, src_tokenizer, trg_tokenizer, max_len=64):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_tokens = self.src_tokenizer(self.src_texts[idx], 
                                         max_length=self.max_len, 
                                         padding='max_length', 
                                         truncation=True, 
                                         return_tensors="pt")
        trg_tokens = self.trg_tokenizer(self.trg_texts[idx], 
                                         max_length=self.max_len, 
                                         padding='max_length', 
                                         truncation=True, 
                                         return_tensors="pt")
        
        return {
            'src': src_tokens['input_ids'].squeeze(0),
            'trg': trg_tokens['input_ids'].squeeze(0)
        }

def get_loaders(task="en-hi", batch_size=32):
    if task == "en-hi":
        try:
            dataset = load_dataset("cfilt/iitb-english-hindi", split='train[:10000]')
            src_texts = [x['en'] for x in dataset['translation']]
            trg_texts = [x['hi'] for x in dataset['translation']]
        except Exception:
            src_texts = ["How are you?", "Hello", "What is your name?"] * 1000
            trg_texts = ["आप कैसे हैं?", "नमस्ते", "आपका नाम क्या है?"] * 1000
            
        src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        trg_tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
    elif task == "en-es":
        try:
            dataset = load_dataset("Helsinki-NLP/opus_books", "en-es", split='train[:10000]')
            src_texts = [x['en'] for x in dataset['translation']]
            trg_texts = [x['es'] for x in dataset['translation']]
        except Exception:
            src_texts = ["How are you?", "Hello", "What is your name?"] * 1000
            trg_texts = ["¿Cómo estás?", "Hola", "¿Cómo te llamas?"] * 1000
            
        src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        trg_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    elif task == "en-fr":
        try:
            dataset = load_dataset("Helsinki-NLP/opus_books", "en-fr", split='train[:10000]')
            src_texts = [x['en'] for x in dataset['translation']]
            trg_texts = [x['fr'] for x in dataset['translation']]
        except Exception:
            src_texts = ["How are you?", "Hello", "What is your name?"] * 1000
            trg_texts = ["Comment allez-vous?", "Bonjour", "Quel est votre nom?"] * 1000
            
        src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        trg_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    filtered_src = []
    filtered_trg = []
    for s, t in zip(src_texts, trg_texts):
        if len(s.strip()) > 0 and len(t.strip()) > 0:
            filtered_src.append(s)
            filtered_trg.append(t)
            
    dataset = TranslationDataset(filtered_src, filtered_trg, src_tokenizer, trg_tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader, src_tokenizer, trg_tokenizer
