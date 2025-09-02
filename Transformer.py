import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # 维度(max_len, d_model)

        # 对1维向量(max_len,)使用unsqueeze(1)将形状改为2维向量(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # [10000^(2i/d_model)]^-1 = exp(2i * -ln(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # 把pe维度改成(1, max_len, d_model), 后面要和x相加, x的维度是(batch_size, seq_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)
        # pe(1, max_len, d_model)
        # x(batch_size, seq_len, d_model)
        # 通过 [:, :x.size(1), :] 选取适配当前输入 seq_len 的部分对输入x进行位置编码


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        B, L_q, D = query.shape
        _, L_k, _ = key.shape
        _, L_v, _ = value.shape


        # 相较于reshape, view不会创建新的张量,效率更高. 前提是张量在内存中的地址必须是连续的
        Q = self.W_q(query).view(B, L_q, self.num_heads, self.d_k).transpose(1, 2)  # (B, L, H, d_k) -> (B, H, L, d_k)
        K = self.W_k(key).view(B, L_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, L_v, self.num_heads, self.d_k).transpose(1, 2)

        # Q(B, H, L, d_k)
        # K(B, H, L, d_k) 转置后变成(B, H, d_k, L)
        # scores = Q @ K^T = (B, H, L, L)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, L, L)

        if mask is not None:
            assert isinstance(mask, torch.Tensor)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # softmax对最后一维L进行归一化, 形状还是(B, H, L, L)
        attn = F.softmax(scores, dim=-1)

        # V(B, H, L, d_k)
        output = torch.matmul(attn, V)  # (B, H, L, d_k)

        # 将结果还原回(B, L, D)进行后续计算. transpose会导致tensor不连续,无法使用view(),所以要先进行contiguous操作
        output = output.transpose(1, 2).contiguous().view(B, L_q, D)
        return self.W_o(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))

        # cross attention中Q来自解码器, K和V来自编码器
        x = self.norm2(x + self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.norm3(x + self.ffn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

    def forward(self, x, mask):
        x = self.embed(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        x = self.embed(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        x = self.fc_out(x)
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=5000):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len)
        self.decoder = TransformerDecoder(vocab_size, d_model, num_layers, num_heads, d_ff, max_len)

    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_out = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return output
