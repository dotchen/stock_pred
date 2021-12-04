import torch
from torch import nn
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNStockModel(nn.Module):
    def __init__(self, embedding_matrix, config):
        super().__init__()

        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)

        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)

        self.tweet_encoder = nn.GRU(
            self.embedding_dim, config['tweet_hidden_dim'],
            num_layers=config['num_tweet_rnn_layers'],
            bidirectional=True,
            dropout=config['dropout'],
            batch_first=True, 
        )

        # self.price_encoder = nn.

        self.tweet_hidden_dim = config['tweet_hidden_dim']

    def forward(self, price_hist, price_lens, tweet_hist, tweet_lens):

        B, T_tw, C, N_tw, L_tw = tweet_hist.size()
        device = tweet_hist.device

        # mask_instruct = torch.arange(L, device=init_state.device)[None, :] < instruct_lengths[:, None]
        # nonzero_tweet_mask = torch.arange(L_tw, device=tweet_hist.device)[None,N]

        # Encode tweets
        nonzero_tweet_indices = (tweet_lens > 0)
        nonzero_tweet_hist = tweet_hist[nonzero_tweet_indices]
        nonzero_tweet_lens = tweet_lens[nonzero_tweet_indices]

        nonzero_tweet_hist = self.embedding(nonzero_tweet_hist)

        nonzero_tweet_hist, nonzero_tweet_lens, sorted_nonzero_tweet_hist_indices, _ = sort_batch_by_length(nonzero_tweet_hist, nonzero_tweet_lens)

        nonzero_tweet_hist = pack_padded_sequence(nonzero_tweet_hist, nonzero_tweet_lens, batch_first=True)
        nonzero_tweet_embd, _ = self.tweet_encoder(nonzero_tweet_hist)
        nonzero_tweet_embd, _ = pad_packed_sequence(nonzero_tweet_embd, batch_first=True)
        nonzero_tweet_embd = nonzero_tweet_embd.index_select(0, sorted_nonzero_tweet_hist_indices)

        tweet_embed = torch.zeros((B,T_tw,C,N_tw,nonzero_tweet_embd.size(1),self.tweet_hidden_dim*2), dtype=torch.float32, device=device)
        tweet_embed[nonzero_tweet_indices] = nonzero_tweet_embd

        # Encode prices
        nonzero_price_indices = (price_lens > 0)
        nonzero_price_hist = price_hist[nonzero_price_indices]
        nonzero_price_lens = price_lens[nonzero_price_indices]

        import pdb; pdb.set_trace()

        nonzero_price_hist, nonzero_price_lens, sorted_nonzero_price_hist_indices, _ = sort_batch_by_length(nonzero_price_hist, nonzero_price_lens)
        nonzero_tweet_hist = pack_padded_sequence(nonzero_tweet_hist, nonzero_tweet_lens, batch_first=True)
        nonzero_tweet_embd, _ = self.tweet_encoder(nonzero_tweet_hist)
        nonzero_tweet_embd, _ = pad_packed_sequence(nonzero_tweet_embd, batch_first=True)
        nonzero_tweet_embd = nonzero_tweet_embd.index_select(0, sorted_nonzero_tweet_hist_indices)

        tweet_embed = torch.zeros((B,T_tw,C,N_tw,nonzero_tweet_embd.size(1),self.tweet_hidden_dim*2), dtype=torch.float32, device=device)
        tweet_embed[nonzero_tweet_indices] = nonzero_tweet_embd
