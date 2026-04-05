import torch
import torch.nn as nn
import torch.optim as optim
from models.seq2seq_model import Encoder, Decoder, Seq2Seq
from data_loader import get_loaders
import os
import tqdm

def train_model(task="en-hi", epochs=15):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"🚀 Using MacBook GPU (MPS) for task {task}") if device.type == 'mps' else print(f"Using device: {device} for task {task}")
    
    # Using more realistic data slices for actual training
    loader, src_tokenizer, trg_tokenizer = get_loaders(task=task, batch_size=32)
    
    INPUT_DIM = src_tokenizer.vocab_size
    OUTPUT_DIM = trg_tokenizer.vocab_size
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
    
    model = Seq2Seq(enc, dec, device).to(device)
    
    # NEW: Check if model already exists to resume training
    save_path = f"model_{task}.pt"
    if os.path.exists(save_path):
        print(f"🔄 Resuming from existing model: {save_path}")
        try:
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
        except Exception as e:
            print(f"⚠️ Could not load checkpoint, starting fresh. Error: {e}")
    
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=trg_tokenizer.pad_token_id)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        pbar = tqdm.tqdm(loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(pbar):
            src = batch['src'].to(device)
            trg = batch['trg'].to(device)
            
            optimizer.zero_grad()
            output = model(src, trg)
            
            # output = [batch size, trg len, trg vocab size]
            # trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            # Use [:, 1:] to skip the <sos> token in the target and the corresponding first output
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} loss: {epoch_loss / len(loader)}")
        
    # Save model
    save_path = f"model_{task}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'src_tokenizer': src_tokenizer,
        'trg_tokenizer': trg_tokenizer
    }, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train translation models")
    parser.add_argument('--task', type=str, default='all', choices=['all', 'en-hi', 'en-es', 'en-fr'])
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    
    if args.task == 'all':
        train_model(task="en-hi", epochs=args.epochs)
        train_model(task="en-es", epochs=args.epochs)
        train_model(task="en-fr", epochs=args.epochs)
    else:
        train_model(task=args.task, epochs=args.epochs)
