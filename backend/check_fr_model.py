import torch
path = 'model_en-fr.pt'
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
for k, v in checkpoint['model_state_dict'].items():
    if 'rnn.weight_ih_l0' in k:
        print(f"{k}: {v.shape}")
