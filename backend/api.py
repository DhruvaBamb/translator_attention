from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, BlipProcessor, BlipForConditionalGeneration
from models.seq2seq_model import Encoder, Decoder, Seq2Seq
import os
from PIL import Image
import io

app = FastAPI(title="Encoder-Decoder Architecture API")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Loading Image Captioning model (Pre-trained Encoder-Decoder)
try:
    caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
except Exception as e:
    print(f"Error loading captioning model: {e}")
    caption_processor = None
    caption_model = None

# Model Loading Helper (Mock/Dummy if files not found)
def load_translation_model(path, src_tok_name, trg_tok_name):
    try:
        if not os.path.exists(path):
            return None, AutoTokenizer.from_pretrained(src_tok_name), AutoTokenizer.from_pretrained(trg_tok_name)
        
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        src_tokenizer = checkpoint['src_tokenizer']
        trg_tokenizer = checkpoint['trg_tokenizer']
        state_dict = checkpoint['model_state_dict']
        
        # Auto-detect parameters from state_dict to avoid mismatches
        INPUT_DIM = src_tokenizer.vocab_size
        OUTPUT_DIM = trg_tokenizer.vocab_size
        
        # Infer dimensions from weights
        # encoder.rnn.weight_ih_l0 shape is [4*hid_dim, emb_dim]
        # decoder.rnn.weight_ih_l0 shape is [4*hid_dim, emb_dim]
        try:
            # Finding HID_DIM and N_LAYERS
            # Check weight_hh_l0 to get hidden_dim (rnn.weight_hh_l0 has shape [4*hid_dim, hid_dim])
            hh_weight = state_dict['encoder.rnn.weight_hh_l0']
            HID_DIM = hh_weight.shape[1]
            
            # Check embedding weight
            emb_weight = state_dict['encoder.embedding.weight']
            ENC_EMB_DIM = emb_weight.shape[1]
            DEC_EMB_DIM = state_dict['decoder.embedding.weight'].shape[1]
            
            # Count layers
            N_LAYERS = 0
            while f'encoder.rnn.weight_ih_l{N_LAYERS}' in state_dict:
                N_LAYERS += 1
                
            print(f"Loading {path} with auto-detected: hid={HID_DIM}, layers={N_LAYERS}, emb={ENC_EMB_DIM}")
        except Exception as e:
            print(f"Warning: Could not auto-detect params for {path}, using defaults. Error: {e}")
            ENC_EMB_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS = 128, 128, 256, 2
        
        enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, 0.5)
        dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, 0.5)
        model = Seq2Seq(enc, dec, torch.device('cpu'))
        
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Failed to load weights due to architecture mismatch (likely Attention upgrade). Ignoring weights. Error: {e}")
            return None, AutoTokenizer.from_pretrained(src_tok_name), AutoTokenizer.from_pretrained(trg_tok_name)
            
        model.eval()
        
        return model, src_tokenizer, trg_tokenizer
    except Exception as e:
        print(f"Error loading model at {path}: {e}")
        return None, AutoTokenizer.from_pretrained(src_tok_name), AutoTokenizer.from_pretrained(trg_tok_name)

# Model instances
# Actually, we load on demand for resource efficiency or keep them in mem
models = {
    "hi": load_translation_model("model_en-hi.pt", "bert-base-uncased", "google/muril-base-cased"),
    "es": load_translation_model("model_en-es.pt", "bert-base-uncased", "bert-base-multilingual-cased"),
    "fr": load_translation_model("model_en-fr.pt", "bert-base-uncased", "bert-base-multilingual-cased")
}

@app.post("/translate/{lang}")
async def translate(lang: str, request: TranslationRequest):
    if lang not in ["hi", "es", "fr"]:
        raise HTTPException(status_code=400, detail="Unsupported language")
    
    model, src_tok, trg_tok = models[lang]
    
    # Simple Inference Logic
    if model is None:
        # Mock translation if no model trained yet
        translations = {
            "hi": {"hello": "नमस्ते", "how are you": "आप कैसे हैं?", "world": "दुनिया"},
            "es": {"hello": "hola", "how are you": "¿cómo estás?", "world": "mundo"},
            "fr": {"hello": "bonjour", "how are you": "comment allez-vous?", "world": "monde"}
        }
        words = request.text.lower().split()
        translated_words = [translations[lang].get(w, w) for w in words]
        return {"translated_text": " ".join(translated_words) + " (Mock Model Output)"}
    
    # Real Model Inference (Better logic for Seq2Seq)
    tokens = src_tok(request.text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        # Encode
        encoder_outputs, hidden, cell = model.encoder(tokens)
        
        # Start token
        sos_id = trg_tok.bos_token_id if trg_tok.bos_token_id else (trg_tok.cls_token_id if trg_tok.cls_token_id else 0)
        eos_id = trg_tok.eos_token_id if trg_tok.eos_token_id else (trg_tok.sep_token_id if trg_tok.sep_token_id else 1)
        pad_id = trg_tok.pad_token_id if trg_tok.pad_token_id is not None else -1
        
        curr_input = torch.tensor([sos_id])
        result = []
        
        # Decode step by step
        print(f"Starting inference for {lang}. SOS={sos_id}, EOS={eos_id}, PAD={pad_id}")
        for i in range(128): # max 128 tokens
            prediction, hidden, cell = model.decoder(curr_input, hidden, cell, encoder_outputs)
            next_token = prediction.argmax(1)
            token_id = next_token.item()
            
            # Print token for debugging
            if i < 10: # Only print first 10 for log brevity 
                 print(f"Step {i}: Predicted token {token_id} ('{trg_tok.decode([token_id])}')")
            
            # Stop if EOS or PAD
            if token_id == eos_id or token_id == pad_id:
                print(f"Inference stopped at step {i} (Found EOS/PAD)")
                break
            
            # Simple repetition prevention (if last 5 tokens are same, stop)
            if len(result) > 5 and all(x == token_id for x in result[-5:]):
                print(f"Inference stopped at step {i} (Repetition detected)")
                break
                
            result.append(token_id)
            curr_input = next_token
        
        translated_text = trg_tok.decode(result, skip_special_tokens=True)
        print(f"Final Translated Result: '{translated_text}'")
        return {"translated_text": translated_text if translated_text else "No output received."}

@app.post("/summarize")
async def summarize(request: SummaryRequest):
    if summ_model is None:
        return {"summary": "Summarization model not loaded. Please ensure internet is available or check backend logs."}
    
    inputs = summ_tokenizer.encode("summarize: " + request.text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = summ_model.generate(inputs, max_length=150, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = summ_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}

@app.post("/caption")
async def caption_image(file: UploadFile = File(...)):
    if caption_model is None:
        # Mock captioning for demonstration
        return {"caption": "A beautiful landscape with green mountains and a blue sky (Mock Output - Model Loading Failed)."}
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        inputs = caption_processor(image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        
        return {"caption": caption}
    except Exception as e:
        print(f"Error processing image: {e}")
        return {"caption": "Error processing image: " + str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
