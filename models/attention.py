import torch
from torch import nn
from allennlp.nn.util import masked_softmax

class SelfAttention(nn.Module):
    def __init__(self, d_dim, k_dim, v_dim):
        super().__init__()

        self.linear_q = nn.Linear(d_dim, k_dim)
        self.linear_k = nn.Linear(d_dim, k_dim)

        self.d_dim = d_dim

        # self.reduce = reduction
        # self.linear_v = nn.Linear(d_dim, v_dim)

    def forward(self, x, mask=None):

        B, L, D = x.size()

        x = x + pos_embd(L, self.d_dim, device=x.device)
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = x
        # v = self.linear_v(x)

        return scaled_dot_product(q, k, v, mask=mask)



class CrossAttention(nn.Module):
    def __init__(self, k_dim, v_dim, pos_embd=False):
        super().__init__()

        self.linear_q = nn.Linear(k_dim, k_dim)
        self.linear_k = nn.Linear(v_dim, k_dim)
        # self.attn_weights = nn.Linear(3*d_dim, 1)

    def forward(self, q, v, mask=None):

        return scaled_dot_product(self.linear_q(q), self.linear_k(v), v, mask=mask).squeeze(1)

def pos_embd(
    seq_len, dim_model, device: torch.device = torch.device("cpu"),
):
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model)

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def scaled_dot_product(query, key, value, mask=None):

    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    if mask is None:
        softmax = torch.softmax(temp / scale, dim=-1)
    else:
        softmax = masked_softmax(temp / scale, mask, dim=-1)

    return softmax.bmm(value)
