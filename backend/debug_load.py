import torch
import os
import sys

# Add models to path
sys.path.append(os.path.abspath('.'))
from models.seq2seq_model import Encoder, Decoder, Seq2Seq

path = 'model_en-es.pt'
try:
    print(f"Attempting to load {path}")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    print("Checkpoint loaded successfully")
    src_tokenizer = checkpoint['src_tokenizer']
    trg_tokenizer = checkpoint['trg_tokenizer']
    print("Tokenizers extracted")
    
    INPUT_DIM = src_tokenizer.vocab_size
    OUTPUT_DIM = trg_tokenizer.vocab_size
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    HID_DIM = 256
    N_LAYERS = 2
    
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, 0.5)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, 0.5)
    model = Seq2Seq(enc, dec, torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model initialized and weights loaded")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
