from torch import optim
from .vanilla import VanillaModel
from .vanilla_sa import VanillaSAModel
from .tweet_only import TweetOnlyModel
from .tweet_only_attn import TweetOnlyAttnModel
from .tweet_vanilla import TweetVanillaModel
from .tweet_price import TweetPriceModel
from .tweet_price_sa import TweetPriceSAModel

def load_model(embedding_matrix, config):
    if config['model'] == 'vanilla':
        return VanillaModel(embedding_matrix, config)
    elif config['model'] == 'vanilla_sa':
        return VanillaSAModel(embedding_matrix, config)
    elif config['model'] == 'tweet_only':
        return TweetOnlyModel(embedding_matrix, config)
    elif config['model'] == 'tweet_only_attn':
        return TweetOnlyAttnModel(embedding_matrix, config)
    elif config['model'] == 'tweet_price':
        return TweetPriceModel(embedding_matrix, config)
    elif config['model'] == 'tweet_price_sa':
        return TweetPriceSAModel(embedding_matrix, config)
    else:
        raise NotImplementedError

def load_optimizer(model, config):
    if config['optim'] == 'sgd':
        return optim.SGD(model.parameters(), lr=float(config['lr']), momentum=0.9, weight_decay=float(config['weight_decay']))
    elif config['optim'] == 'adam':
        return optim.Adam(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    elif config['optim'] == 'adamw':
        return optim.AdamW(model.parameters(), lr=float(config['lr']), weight_decay=float(config['weight_decay']))
    else:
        return NotImplementedError
