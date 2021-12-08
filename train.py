import yaml
import tqdm
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.nn import functional as F
from models import load_model, load_optimizer
from data import load_data, load_embeddings
from utils import metrics, save_model, random_seed, to_numpy

def train(model, data, optimizer, log_callback, device=torch.device('cuda')):

    # accs, macro_f1s = [], []
    preds = []
    gts = []

    prefix = 'Eval' if optimizer is None else 'Train'

    for price_hist, price_lens, tweet_hist, tweet_lens, shift in tqdm.tqdm(data, desc=prefix):

        price_hist = price_hist.to(device)
        price_lens = price_lens.to(device)
        tweet_hist = tweet_hist.to(device)
        tweet_lens = tweet_lens.to(device)
        shift = shift.to(device)

        pred_shift = model(price_hist, price_lens, tweet_hist, tweet_lens)

        preds.extend(list(to_numpy(pred_shift.argmax(dim=1).flatten())))
        gts.extend(list(to_numpy(shift.flatten())))

        # acc, macro_f1 = metrics(pred_shift.argmax(dim=1), shift)
        # accs.append(acc)
        # macro_f1s.append(macro_f1)

        if optimizer is not None:
            loss = F.cross_entropy(pred_shift, shift, weight=torch.tensor([1.1,1,1],dtype=torch.float32, device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log_callback(float(loss))

    print (np.unique(preds, return_counts=True))
    print (np.unique(gts, return_counts=True))

    log_callback(None, acc=(np.array(preds)==np.array(gts)).mean(), preds=preds, gts=gts)


def main(args):

    random_seed(42)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    print ("Loading data...", flush=True)
    train_data, valid_data, train_vocab = load_data(config)

    embedding_matrix = load_embeddings(config['glove_path'], train_vocab)

    device = torch.device('cuda' if args.cuda else 'cpus')

    print ("Loading model...", flush=True)
    model = load_model(embedding_matrix, config)
    model = model.to(device)
    model.vocab = train_vocab

    # Init wandb
    wandb.init(config=config)
    wandb.watch(model, log_freq=10)

    # Start training
    optimizer = load_optimizer(model, config)

    global_it = 0
    def get_callback(epoch, prefix):

        def callback(loss, acc=None, preds=None, gts=None):
            if prefix == 'valid' and acc is None:
                return

            nonlocal global_it
            if acc is None and preds is None and gts is None:
                global_it += 1
                if global_it % args.log_freq == 0:
                    wandb.log({'it': global_it, 'loss': loss})
            else:
                # import pdb; pdb.set_trace()
                plt.imshow(confusion_matrix(gts, preds))
                wandb.log({
                    'epoch': epoch, 
                    f'{prefix}/acc': acc,
                    f'{prefix}/confusion_matrix': wandb.Image(plt),
                    f'{prefix}/gt': wandb.Histogram(gts),
                    f'{prefix}/pred': wandb.Histogram(preds),
                })
                plt.close('all')

        return callback

    save_dir = wandb.run.dir
    for epoch in range(args.num_epochs):
        train(model, train_data, optimizer, device=device, log_callback=get_callback(epoch+1, 'train'))
        
        model.eval()
        with torch.no_grad():
            train(model, valid_data, None, device=device, log_callback=get_callback(epoch+1, 'valid'))
        model.train()

        # Save model
        saved_path = save_model(model, epoch, wandb.run.dir)
        wandb.save(saved_path)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', default='config.yaml')
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--log-freq', type=int, default=10)

    parser.add_argument('--cpu', dest='cuda', action='store_false', default=True)
    # parser.add_argument('--batch-size', type=int, default=4)
    # parser.add_argument('--weight-decay', type=float, default=1e-4)

    # parser.add_argument('--glove-path', default='glove/glove.twitter.27B.100d.txt')
    args = parser.parse_args()
    
    main(args)
