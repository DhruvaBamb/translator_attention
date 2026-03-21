from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from models.seq2seq_model import Encoder, Decoder, Seq2Seq
import os

app = FastAPI(title="Encoder-Decoder Architecture API")

class TranslationRequest(BaseModel):
    text: str

class SummaryRequest(BaseModel):
    text: str

# Loading Summary model (Pre-trained for efficiency)
try:
    summ_tokenizer = T5Tokenizer.from_pretrained("t5-small")
    summ_model = T5ForConditionalGeneration.from_pretrained("t5-small")
except Exception as e:
    print(f"Error loading summary model: {e}")
    summ_model = None

# Model Loading Helper (Mock/Dummy if files not found)
def load_translation_model(path, src_tok_name, trg_tok_name):
    try:
        if not os.path.exists(path):
            return None, AutoTokenizer.from_pretrained(src_tok_name), AutoTokenizer.from_pretrained(trg_tok_name)
        
        checkpoint = torch.load(path, map_location='cpu')
        src_tokenizer = checkpoint['src_tokenizer']
        trg_tokenizer = checkpoint['trg_tokenizer']
        
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
        
        return model, src_tokenizer, trg_tokenizer
    except Exception as e:
        print(f"Error loading model at {path}: {e}")
        return None, AutoTokenizer.from_pretrained(src_tok_name), AutoTokenizer.from_pretrained(trg_tok_name)

# Model instances
# Actually, we load on demand for resource efficiency or keep them in mem
models = {
    "hi": load_translation_model("model_en-hi.pt", "bert-base-uncased", "google/muril-base-cased"),
    "es": load_translation_model("model_en-es.pt", "bert-base-uncased", "bert-base-multilingual-cased")
}

@app.post("/translate/{lang}")
async def translate(lang: str, request: TranslationRequest):
    if lang not in ["hi", "es"]:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    model, src_tok, trg_tok = models[lang]
    
    # Simple Inference Logic
    if model is None:
        # Mock translation if no model trained yet
        translations = {
            "hi": {"hello": "नमस्ते", "how are you": "आप कैसे हैं?", "world": "दुनिया"},
            "es": {"hello": "hola", "how are you": "¿cómo estás?", "world": "mundo"}
        }
        words = request.text.lower().split()
        translated_words = [translations[lang].get(w, w) for w in words]
        return {"translated_text": " ".join(translated_words) + " (Mock Model Output)"}
    
    # Real Model Inference
    tokens = src_tok(request.text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        # trg = <sos>
        trg = torch.tensor([[trg_tok.bos_token_id if trg_tok.bos_token_id else 0]]).repeat(tokens.size(0), 1)
        # Sequence generation loop (greedy)
        for _ in range(32): # max len 32
            output = model(tokens, trg)
            next_token = output[:, -1, :].argmax(1).unsqueeze(1)
            trg = torch.cat([trg, next_token], dim=1)
            if next_token.item() == (trg_tok.eos_token_id if trg_tok.eos_token_id else 1):
                break
        
        translated_text = trg_tok.decode(trg[0], skip_special_tokens=True)
        return {"translated_text": translated_text}

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if summ_model is None:
        return {"summary": "Summarization model not loaded. Please ensure internet is available or check backend logs."}
    
    inputs = summ_tokenizer.encode("summarize: " + request.text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summ_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
