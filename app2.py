import torch
import torch.nn as nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = self.dropout(torch.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.attention = ScaledDotProductAttention(temperature=math.sqrt(d_k))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q + residual)

        return q, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.w_2(self.dropout(torch.relu(self.w_1(x))))
        x = self.layer_norm(x + residual)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x, attn = self.attention(x, x, x, mask)
        x = self.ffn(x)
        return x, attn

class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head, d_k, d_v, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

    def forward(self, dec_input, enc_output, self_mask=None, enc_dec_mask=None):
        dec_output, self_attn = self.self_attention(dec_input, dec_input, dec_input, self_mask)
        dec_output, enc_dec_attn = self.enc_dec_attention(dec_output, enc_output, enc_output, enc_dec_mask)
        dec_output = self.ffn(dec_output)
        return dec_output, self_attn, enc_dec_attn

class Transformer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, d_k, d_v, n_layers, n_classes, max_len=5000, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, d_ff, n_head, d_k, d_v, dropout)
            for _ in range(n_layers)
        ])
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, enc_dec_mask=None):
        src = self.embedding(src)
        src = self.positional_encoding(src)
        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)

        enc_output = src
        for layer in self.encoder_layers:
            enc_output, _ = layer(enc_output, src_mask)

        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output, _, _ = layer(dec_output, enc_output, tgt_mask, enc_dec_mask)

        output = self.fc(dec_output.mean(dim=1))
        return output
    

if __name__ == "__main__":
    # Parameters for the Transformer model
    d_model = 512
    d_ff = 2048
    n_head = 8
    d_k = 64
    d_v = 64
    n_layers = 6
    n_classes = 10
    max_len = 5000
    dropout = 0.1

    # Instantiate the Transformer model
    model = Transformer(d_model, d_ff, n_head, d_k, d_v, n_layers, n_classes, max_len, dropout)

    # Example input data
    src = torch.randint(0, max_len, (32, 100))  # Source sequence: batch_size=32, seq_len=100
    tgt = torch.randint(0, max_len, (32, 80))   # Target sequence: batch_size=32, seq_len=80

    # Forward pass through the model
    output = model(src, tgt)

    # Print the output shape
    print(output.shape)  # should be (32, 10), representing the class scores for each sequence in the batch