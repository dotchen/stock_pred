from torch import nn

class RNNStockModel(nn.Module):
    def __init__(self, embedding_matrix, config):
        super().__init__()

        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)

        # Load our embedding matrix weights into the Embedding object,
        # and make them untrainable (requires_grad=False)
        # TODO: Your code here.
        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)


        # self.tweet_encoder = nn.GRU()

    def forward(self, price_hist, price_lens, tweet_hist, tweet_lens):
        pass