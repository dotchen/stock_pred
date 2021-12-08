import torch
from torch import nn
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .basic import BasicModel
from .attention import SelfAttention


class VanillaSAModel(BasicModel):
    def __init__(self, embedding_matrix, config):
        super().__init__(embedding_matrix, config)

        # TODO: make this argument
        self.price_encoder = nn.GRU(
            3, config['price_hidden_dim'],
            num_layers=config['num_price_rnn_layers'],
            dropout=config['dropout'],
            batch_first=True
        )

        self.price_hidden_dim = config['price_hidden_dim']

        self.price_sa = SelfAttention(self.price_hidden_dim, 16, self.price_hidden_dim)
        self.predict = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(self.price_hidden_dim, len(config['price_shifts_bins'])+1)
        )

    def forward(self, price_hist, price_lens, tweet_hist, tweet_lens):

        B, C, T_pr, _ = price_hist.size()

        device = price_hist.device

        # Encode prices
        price_embd = torch.zeros((B,C,self.price_hidden_dim), dtype=torch.float32, device=device)

        nonzero_price_indices = (price_lens > 0)
        nonzero_price_hist = price_hist[nonzero_price_indices].view(-1,T_pr,3)
        nonzero_price_lens = price_lens[nonzero_price_indices][:,None].repeat(1,85).view(-1)

        if nonzero_price_lens.sum() > 0:

            nonzero_price_hist, nonzero_price_lens, sorted_nonzero_price_hist_indices, _ = sort_batch_by_length(nonzero_price_hist, nonzero_price_lens)
            nonzero_price_hist = pack_padded_sequence(nonzero_price_hist, nonzero_price_lens, batch_first=True)
            _, nonzero_price_embd = self.price_encoder(nonzero_price_hist)

            nonzero_price_embd = nonzero_price_embd[0].index_select(0, sorted_nonzero_price_hist_indices)

            price_embd[nonzero_price_indices] = nonzero_price_embd.view(-1,C,self.price_hidden_dim)

        price_embd = self.price_sa(price_embd)
        return self.predict(price_embd).permute(0,2,1)
