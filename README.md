# UT Austin CS388 (Fall 2021) Final Project: Dian Chen
Stock price movement prediction with Tweets and Prices.

## Install
First install conda and create a dedicated python environment.
`conda create -n cs388 python=3.6`,

Then in that environment, install `PyTorch`, `pyyaml` and `wandb` either through `conda` or `pip`.

Also download the [dataset](https://github.com/yumoxu/stocknet-dataset/).

## Train Models
Edit `config.yaml` with hyperparameter and model configs of choice and then do 
```bash
python train.py
```
The best one I get was `model: tweet_price_sa` with default parameters which gets about 38% test set accuracies.
