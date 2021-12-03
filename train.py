import yaml
import tqdm
import torch

from models import load_model, load_optimizer
from data import load_data, load_embeddings
# from models impor

def train(model, data, optimizer, device=torch.device('cuda')):

    if optimizer is None:
        prefix = 'eval'
        do_train = False
    else:
        pretix = 'train'
        do_train = True

    for price_hist, price_lens, tweet_hist, tweet_lens, shift in tqdm.tqdm(data):

        price_hist = price_hist.to(device)
        price_lens = price_lens.to(device)
        tweet_hist = tweet_hist.to(device)
        tweet_lens = tweet_lens.to(device)

        pred_shift = model(price_hist, price_lens, tweet_hist, tweet_lens)

def main(args):

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    print ("Loading data...", flush=True)
    train_data, valid_data, train_vocab = load_data(config)

    embedding_matrix = load_embeddings(config['glove_path'], train_vocab)

    device = torch.device('cuda' if args.cuda else 'cpus')

    print ("Loading model...", flush=True)
    model = load_model(embedding_matrix, config)
    model = model.to(device)

    # Start training
    optimizer = load_optimizer(model, config)
    
    for epoch in range(args.num_epochs):
        train(model, train_data, optimizer, device=device)
        with torch.no_grad():
            train(model, valid_data, None, device=device)

        


if __name__ == '__main__':

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--num-epochs', type=int, default=10)

    parser.add_argument('--cpu', dest='cuda', action='store_false', default=True)
    # parser.add_argument('--batch-size', type=int, default=4)
    # parser.add_argument('--weight-decay', type=float, default=1e-4)

    # parser.add_argument('--glove-path', default='glove/glove.twitter.27B.100d.txt')
    args = parser.parse_args()
    
    main(args)
