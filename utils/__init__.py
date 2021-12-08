import os
import numpy as np
import torch
import random

def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

def metrics(pred, targ, num_classes=3):

    pred = to_numpy(pred)
    targ = to_numpy(targ)

    acc = np.mean(pred==targ)
    macro_f1 = np.mean([f1(pred, targ, clas=clas) for clas in range(num_classes)])
    # macros = [f1(pred, targ, clas=clas) for clas in range()]

    return acc, macro_f1

def f1(pred, targ, clas):

    tp = float((targ==clas).sum())
    fp = float((pred==clas).sum())
    fn = float((pred!=clas).sum())

    return float(2*tp / (2*tp + fp + fn))

def to_numpy(x):
    return x.detach().cpu().numpy()


def save_model(model, epoch, save_dir):
    save_path = os.path.join(save_dir, f'model_{epoch}.th')
    torch.save(model.state_dict(), save_path)
    return save_path