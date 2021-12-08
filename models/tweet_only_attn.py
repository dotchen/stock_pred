import torch
from torch import nn
from allennlp.nn.util import sort_batch_by_length
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .basic import BasicModel
from .attention import SelfAttention, CrossAttention

class TweetOnlyAttnModel(BasicModel):
    def __init__(self, embedding_matrix, config):
        super().__init__(embedding_matrix, config)

        self.tweet_encoder = nn.GRU(
            self.embedding_dim, config['tweet_hidden_dim'],
            num_layers=config['num_tweet_rnn_layers'],
            # bidirectional=True,
            dropout=config['dropout'],
            batch_first=True, 
        )

        # TODO: make this argument
        # self.price_encoder = nn.GRU(
        #     3, config['price_hidden_dim'],
        #     num_layers=config['num_price_rnn_layers'],
        #     dropout=config['dropout'],
        #     batch_first=True
        # )

        self.tweet_hidden_dim = config['tweet_hidden_dim']
        self.price_hidden_dim = config['price_hidden_dim']

        # self.tweet_sa = SelfAttention(self.tweet_hidden_dim, 32, 32)
        self.tweet_query = nn.Parameter(torch.empty((1,self.tweet_hidden_dim), dtype=torch.float32), requires_grad=True)
        nn.init.xavier_uniform_(self.tweet_query)
        self.tweet_attn = CrossAttention(self.tweet_hidden_dim, self.tweet_hidden_dim)
        # self.tweet_sa   = SelfAttention(self.tweet_hidden_dim, 32, self.tweet_hidden_dim)

        # self.predict = nn.Linear(self.embedding_dim, len(config['price_shifts_bins'])+1)

        self.predict = nn.Linear(self.tweet_hidden_dim, len(config['price_shifts_bins'])+1)

    def forward(self, price_hist, price_lens, tweet_hist, tweet_lens):

        B, T_tw, C, N_tw, L_tw = tweet_hist.size()
        _,  _, T_pr, _ = price_hist.size()

        device = tweet_hist.device

        # Encode tweets
        nonzero_tweet_indices = (tweet_lens > 0)
        nonzero_tweet_hist = tweet_hist[nonzero_tweet_indices]
        nonzero_tweet_lens = tweet_lens[nonzero_tweet_indices]

        if nonzero_tweet_lens.sum() > 0:

            # print (self.get_sentence(nonzero_tweet_hist[0]))

            nonzero_tweet_hist = self.embedding(nonzero_tweet_hist)

            nonzero_tweet_hist, nonzero_tweet_lens, sorted_nonzero_tweet_hist_indices, _ = sort_batch_by_length(nonzero_tweet_hist, nonzero_tweet_lens)


            nonzero_tweet_hist = pack_padded_sequence(nonzero_tweet_hist, nonzero_tweet_lens, batch_first=True)
            _, nonzero_tweet_embd = self.tweet_encoder(nonzero_tweet_hist)
            # nonzero_tweet_embd, _ = pad_packed_sequence(nonzero_tweet_embd, batch_first=True)
            nonzero_tweet_embd = nonzero_tweet_embd[0].index_select(0, sorted_nonzero_tweet_hist_indices)

            tweet_embd = torch.zeros((B,T_tw,C,N_tw,self.tweet_hidden_dim), dtype=torch.float32, device=device)
            tweet_embd[nonzero_tweet_indices] = nonzero_tweet_embd

        else:
            tweet_embd = torch.zeros((B,T_tw,C,N_tw,self.tweet_hidden_dim), dtype=torch.float32, device=device)

        # # Encode prices
        # price_embd = torch.zeros((B,C,self.price_hidden_dim), dtype=torch.float32, device=device)

        # nonzero_price_indices = (price_lens > 0)
        # nonzero_price_hist = price_hist[nonzero_price_indices].view(-1,T_pr,3)
        # nonzero_price_lens = price_lens[nonzero_price_indices][:,None].repeat(1,85).view(-1)

        # if nonzero_price_lens.sum() > 0:

        #     nonzero_price_hist, nonzero_price_lens, sorted_nonzero_price_hist_indices, _ = sort_batch_by_length(nonzero_price_hist, nonzero_price_lens)
        #     nonzero_price_hist = pack_padded_sequence(nonzero_price_hist, nonzero_price_lens, batch_first=True)
        #     _, nonzero_price_embd = self.price_encoder(nonzero_price_hist)

        #     nonzero_price_embd = nonzero_price_embd[0].index_select(0, sorted_nonzero_price_hist_indices)

        #     price_embd[nonzero_price_indices] = nonzero_price_embd.view(-1,C,self.price_hidden_dim)
        # price_embd = price_embd.view(B,C,self.price_hidden_dim)

        # Process tweets
        # import pdb; pdb.set_trace()

        values = tweet_embd.view(-1,N_tw,self.tweet_hidden_dim)
        queries = self.tweet_query[None].expand(values.size(0), 1, self.tweet_hidden_dim)

        tweet_embd = self.tweet_attn(
            queries, values,
            mask=(tweet_lens>0).view(-1,N_tw)
        ).view(B,T_tw,C,-1)

        # tweet_embd = tweet_embd
        # import pdb; pdb.set_trace()
        mean_tweet_embd = tweet_embd.mean(dim=1)
        # tweet_embd = tweet_embd.view(B,N_tw,-1)
        # import pdb; pdb.set_trace()
        # mean_tweet_embd = tweet_embd.sum(dim=-2) / (1e-7+tweet_lens).unsqueeze(-1)   # Average over tokens
        # mean_tweet_embd = tweet_embd.mean(dim=1)                                     # Average over time
        # mean_tweet_embd = mean_tweet_embd.mean(dim=2)                                # Average over tweets

        # Process cross-stocks relationships
        # embd = torch.cat([mean_tweet_embd, price_embd], dim=-1)

        return self.predict(mean_tweet_embd).permute(0,2,1)

        # mean_price_embd = price_embd.mean(dim=1)
    
    def get_sentence(self, x):

        return [self.vocab.get_token_from_index(int(t)) for t in x]


