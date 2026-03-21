import torch
import torch.nn as nn
import torch.optim as optim
from models.seq2seq_model import Encoder, Decoder, Seq2Seq
from data_loader import get_loaders
import os
import tqdm

def train_min(task="en-hi", limit=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Minimal training for {task}...")
    
    loader, src_tokenizer, trg_tokenizer = get_loaders(task=task, batch_size=2)
    
    INPUT_DIM = src_tokenizer.vocab_size
    OUTPUT_DIM = trg_tokenizer.vocab_size
    model = Seq2Seq(
        Encoder(INPUT_DIM, 64, 128, 2, 0.5),
        Decoder(OUTPUT_DIM, 64, 128, 2, 0.5),
        device
    ).to(device)
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tokenizer.pad_token_id)
    
    model.train()
    for i, batch in enumerate(loader):
        if i >= limit: break
        src = batch['src'].to(device)
        trg = batch['trg'].to(device)
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output[1:].view(-1, OUTPUT_DIM), trg[1:].view(-1))
        loss.backward()
        optimizer.step()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_tokenizer': src_tokenizer,
        'trg_tokenizer': trg_tokenizer
    }, f"model_{task}.pt")
    print(f"Model saved for {task}")

if __name__ == "__main__":
    train_min("en-hi")
    train_min("en-es")
