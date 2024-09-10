from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
import tiktoken


# using data class , we don't have to write the whole class, it wrie many functions automatically
@dataclass
class GPT2Config:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y



#Transformer Block
class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd)  ,#token embeddings  total_vocab_length x embd_size (for each token , a corresponding embedding vector of size embd_size)
            wpe = nn.Embedding(config.block_size, config.n_embd),#positional embeddings, max_sequence length x embd_size
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]), # transformer block embeddings
            ln_f = nn.LayerNorm(config.n_embd), 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # embd_size x vocab size
        self.encoder = tiktoken.get_encoding("gpt2")

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {"n_layer":12, "n_head":12, "n_embd":768,"vocab_size":50257,"block_size":1024}
        # create a from-scratch initialized minGPT model
        config = GPT2Config(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    def forward(self, idx):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits
    
    def encode_tiktoken(self,input:str):
        encoded = self.encoder.encode(input)
        tokens= torch.tensor(encoded, dtype=torch.long)
        return tokens

    def generate(self,x,max_gen_length):
        while x.size(1) < max_gen_length:
            with torch.no_grad():
                logits = model(x) # logits are model outputs before softmax
                logits = logits[:,-1,:] #just keeping last token , size = B * emd_size
                probs = F.softmax(logits,dim =-1) # take softmax across last dimension (embeddingsize dim)
                # doing top 50 sampling, sets all tokens with less than 0.5 prob to 0, rare tokens are not sampled.
                topk_probs, top_indices = torch.topk(probs,50,dim=-1)
                #sampling
                sample_indices = torch.multinomial(topk_probs,1)
                samples = torch.gather(top_indices,-1,sample_indices) #size = B x 1
                x = torch.cat((x,samples),dim=1) # concatenating sampled token
        
        tokens = x[:max_gen_length].tolist()
        return tokens
    
    def decode_tiktoken(self,batch_size,tokens):
        decoded_sent = []
        for i in range(batch_size):
            decoded_sent.append(self.encoder.decode(tokens[i]))
        return decoded_sent


    
if __name__ == "__main__":
    model = GPT.from_pretrained("gpt2")
    num_sent = 4
    max_gen_length = 60
    # not training model
    model.eval()
    if torch.cuda.is_available():
        device = "gpu"
    else:
        device = "cpu"
    
    model.to(device) # transferring model and all its weights to gpu
    input = "A data scientist's work is to "
    tokens = model.encode_tiktoken(input)
    tokens = tokens.unsqueeze(0).repeat(num_sent,1) # unsqueeze adds 1 dimension at 0, then repeat 5 times
    tokens.to(device)
    tokens = model.generate(tokens,max_gen_length)
    decoded = model.decode_tiktoken(num_sent,tokens)
    for sent in decoded:
        print(">",sent)
    









