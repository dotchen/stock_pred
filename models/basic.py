from torch import nn

class BasicModel(nn.Module):
    def __init__(self, embedding_matrix, config):
        super().__init__()

        self.embedding_matrix = embedding_matrix
        self.num_embedding_words = embedding_matrix.size(0)
        self.embedding_dim = embedding_matrix.size(1)

        self.embedding = nn.Embedding(self.num_embedding_words,
                                      self.embedding_dim, padding_idx=0)

        self.embedding.weight = nn.Parameter(self.embedding_matrix,
                                             requires_grad=False)
