import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import StockNetDataset

def load_data(config):
    train_dataset = StockNetDataset(config, date_range=['2014-01-01', '2015-7-31'])
    valid_dataset = StockNetDataset(config, companies=train_dataset.companies, vocabs=train_dataset.vocabs, date_range=['2015-08-01', '2016-01-01'])

    train_data = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    valid_data = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        drop_last=False
    )

    return train_data, valid_data, train_dataset.vocabs





def load_embeddings(glove_path, vocab):
    """
    Create an embedding matrix for a Vocabulary.
    """
    vocab_size = vocab.get_vocab_size()
    words_to_keep = set(vocab.get_index_to_token_vocabulary().values())
    glove_embeddings = {}
    embedding_dim = None

    with open(glove_path) as glove_file:
        for line in glove_file:
            fields = line.strip().split(" ")
            word = fields[0]
            if word in words_to_keep:
                vector = np.asarray(fields[1:], dtype="float32")
                if embedding_dim is None:
                    embedding_dim = len(vector)
                else:
                    assert embedding_dim == len(vector)
                glove_embeddings[word] = vector

    all_embeddings = np.asarray(list(glove_embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    embedding_matrix = torch.FloatTensor(
        vocab_size, embedding_dim).normal_(
            embeddings_mean, embeddings_std)
    # Manually zero out the embedding of the padding token (0).
    embedding_matrix[0].fill_(0)
    # This starts from 1 because 0 is the padding token, which
    # we don't want to modify.
    for i in range(1, vocab_size):
        word = vocab.get_token_from_index(i)

        # If we don't have a pre-trained vector for this word,
        # we don't change the row and the word has random initialization.
        if word in glove_embeddings:
            embedding_matrix[i] = torch.FloatTensor(glove_embeddings[word])
    return embedding_matrix
