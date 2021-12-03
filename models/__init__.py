from torch import optim
from .rnn import RNNStockModel


def load_model(embedding_matrix, config):
    if config['model'] == 'rnn':
        return RNNStockModel(embedding_matrix, config)
    else:
        raise NotImplementedError

def load_optimizer(model, config):
    if config['optim'] == 'adam':
        return optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    else:
        return NotImplementedError

