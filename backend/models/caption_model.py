import torch
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, embed_size):
        super(ImageEncoder, self).__init__()
        # Using pre-trained ResNet-50
        resnet = models.resnet50(pretrained=True)
        # Remove AdaptiveAvgPool2d and Linear layer to keep spatial resolution
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # ResNet50 last conv block has 2048 output channels
        self.embed = nn.Linear(2048, embed_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Extract features
        features = self.resnet(images) # [batch_size, 2048, H, W]
        batch_size = features.size(0)
        
        # Flatten spatial dimensions: [batch_size, 2048, H*W] -> [batch_size, H*W, 2048]
        features = features.view(batch_size, features.size(1), -1).permute(0, 2, 1)
        
        # Project to embedding space
        features = self.embed(features)
        return self.dropout(features)


class CaptionAttention(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super(CaptionAttention, self).__init__()
        self.attn = nn.Linear(embed_size + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, features, hidden):
        # features = [batch_size, num_pixels, embed_size]
        # hidden = [batch_size, hidden_size] (top layer of LSTM)
        
        num_pixels = features.size(1)
        hidden_repeated = hidden.unsqueeze(1).repeat(1, num_pixels, 1)
        
        energy = torch.tanh(self.attn(torch.cat((features, hidden_repeated), dim=2)))
        attention = self.v(energy).squeeze(2)
        alpha = torch.softmax(attention, dim=1)
        
        # context = sum over pixels
        context = (features * alpha.unsqueeze(2)).sum(dim=1) # [batch_size, embed_size]
        
        return context, alpha


class CaptionDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptionDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = CaptionAttention(embed_size, hidden_size)
        
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)
        
        # Networks to create initial hidden state from image features
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def init_hidden(self, features):
        mean_features = features.mean(dim=1)
        h = self.init_h(mean_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c = self.init_c(mean_features).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h, c

    def forward(self, features, captions):
        # captions = [batch_size, seq_len]
        # features = [batch_size, num_pixels, embed_size]
        batch_size = features.size(0)
        seq_len = captions.size(1)
        
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(features.device)
        
        h, c = self.init_hidden(features)
        
        # Embed captions
        embeddings = self.embed(captions)
        
        for t in range(seq_len):
            context, _ = self.attention(features, h[-1])
            
            # embedded_token: [batch_size, 1, embed_size]
            embedded_token = embeddings[:, t, :].unsqueeze(1)
            # context: [batch_size, 1, embed_size]
            context = context.unsqueeze(1)
            
            lstm_input = torch.cat((embedded_token, context), dim=2)
            out, (h, c) = self.lstm(lstm_input, (h, c))
            
            outputs[:, t, :] = self.linear(out.squeeze(1))
            
        return outputs


class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = ImageEncoder(embed_size)
        self.decoder = CaptionDecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, tokenizer, max_length=20):
        # Generate caption for a single image
        result_caption = []
        
        with torch.no_grad():
            features = self.encoder(image) # [1, num_pixels, embed_size]
            h, c = self.decoder.init_hidden(features)
            
            # Use <sos> token as initial input
            bos = tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None else 0
            eos = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None else 1
            
            input_token = torch.tensor([bos]).unsqueeze(0).to(image.device)
            
            for _ in range(max_length):
                context, _ = self.decoder.attention(features, h[-1])
                embedded_token = self.decoder.embed(input_token)
                
                lstm_input = torch.cat((embedded_token, context.unsqueeze(1)), dim=2)
                out, (h, c) = self.decoder.lstm(lstm_input, (h, c))
                
                output = self.decoder.linear(out.squeeze(1))
                predicted = output.argmax(1)
                
                result_caption.append(predicted.item())
                input_token = predicted.unsqueeze(1)
                
                # Check for <eos>
                if predicted.item() == eos:
                    break
        
        return [tokenizer.decode([idx]) for idx in result_caption]
