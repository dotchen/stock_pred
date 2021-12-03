import glob
import json
import datetime
import numpy as np
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset
from allennlp.data.vocabulary import Vocabulary

class StockNetDataset(Dataset):
    def __init__(self, config, vocabs=None, date_range=['2014-07-01','2016-01-01']):
        super().__init__()

        # Load config
        for key, value in config.items():
            setattr(self, key, value)

        # Process dates
        start_year, start_month, start_day = map(int, date_range[0].split('-'))
        end_year, end_month, end_day = map(int, date_range[1].split('-'))

        start_date = datetime.date(start_year, start_month, start_day)
        end_date = datetime.date(end_year, end_month, end_day)

        # Should remember date as the right order
        self.prices = OrderedDict()
        self.shifts = OrderedDict()
        self.tweets = OrderedDict()
        
        self.shift_counts = defaultdict(int)
        
        self.vocabs = Vocabulary()

        self.max_tweet_len = 0
        self.max_num_tweet = 0

        for stock_data in glob.glob(f'{self.data_dir}/price/preprocessed/**'):
            stock = stock_data.split('/')[-1].split('.txt')[0]
            with open(stock_data) as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                cells = line.split()
                
                date = cells[0]

                year, month, day = map(int, date.split('-'))
                if start_date <= datetime.date(year, month, day) <= end_date:

                    movement_percent, open_price, high_price, low_price, close_price = map(float, cells[1:6])

                    shift_bin = np.searchsorted(self.price_shifts_bins, movement_percent*100)

                    self.maybe_add(date, [self.prices, self.shifts])
                    self.prices[date][stock] = close_price
                    self.shifts[date][stock] = shift_bin
                    
                    self.shift_counts[shift_bin] += 1

        # First pass: build vocabulary
        for stock_data in glob.glob(f'{self.data_dir}/tweet/preprocessed/**'):
            stock = stock_data.split('/')[-1].split('.txt')[0]

            for tweet_data in glob.glob(f'{stock_data}/**'):
                date = tweet_data.split('/')[-1]

                year, month, day = map(int, date.split('-'))
                if start_date <= datetime.date(year, month, day) <= end_date:

                    with open(tweet_data) as f:
                        lines = f.readlines()

                    for line in lines:
                        line = line.strip()
                        tweet_json = json.loads(line)

                        # First pass
                        for token in tweet_json['text']:
                            self.vocabs.add_token_to_namespace(token)

                        # Second pass
                        if vocabs is None:
                            tweet = [self.vocabs.get_token_index(token) for token in tweet_json['text']]
                        else:
                            tweet = [vocabs.get_token_index(token) for token in tweet_json['text']]

                        self.maybe_add(date, [self.tweets], cls_func=lambda: defaultdict(list))
                        self.tweets[date][stock].append(tweet)

                        self.max_tweet_len = max(self.max_tweet_len, len(tweet_json['text']))
    
                    self.max_num_tweet = max(self.max_num_tweet, len(lines))
    
        def day_of_year(date):
            year, month, day = map(int, date.split('-'))
            gap = datetime.date(year, month, day) - start_date
            return gap.days
    

        self.dates = list(sorted(self.shifts.keys()))
        self.tweet_dates = sorted(list(self.tweets.keys()))
        
        self.dates_days = [day_of_year(date) for date in self.dates]
        self.tweet_dates_days = [day_of_year(date) for date in self.tweet_dates]

        self.companies = list(self.shifts[self.dates[0]].keys())

        if vocabs is not None:
            self.vocabs = vocabs

        print (self.shift_counts)

    def __getitem__(self, idx):

        date = self.dates[idx]
        
        # Retrieve prices
        if idx >= self.price_encode_lens:
            price_hist_len = self.price_encode_lens
        else:
            price_hist_len = idx

        price_hist = np.zeros((self.price_encode_lens, len(self.companies)), dtype=np.float32)

        for c, company in enumerate(self.companies):
            for t, _date in enumerate(self.dates[idx-price_hist_len:idx]):
                price_hist[t,c] = self.prices[_date][company]

        # Retrieve tweets
        date_in_days = self.dates_days[idx]

        closest_day_idx = np.searchsorted(self.tweet_dates_days, date_in_days)


        if closest_day_idx >= self.tweet_encode_lens-1:
            tweet_hist_len = self.tweet_encode_lens
        else:
            tweet_hist_len = closest_day_idx+1

        tweet_hist = np.zeros((self.tweet_encode_lens, len(self.companies), self.max_num_tweet, self.max_tweet_len), dtype=np.float32)
        tweet_lens = np.zeros((self.tweet_encode_lens, len(self.companies), self.max_num_tweet), dtype=np.long)

        for t, _date in enumerate(self.tweet_dates[closest_day_idx-tweet_hist_len+1:closest_day_idx+1]):
            for company in self.tweets[_date].keys():
                if company not in self.companies:
                    continue
                c = self.companies.index(company)
                for i, tweet in enumerate(self.tweets[_date][company]):
                    tweet_hist[t,c,i,:len(tweet)] = tweet
                    tweet_lens[t,c,i] = len(tweet)

        # Retrieve price shifts
        shift = self.shifts[date]
        
        return price_hist, price_hist_len, tweet_hist, tweet_lens, shift

        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        # tweet_hist = np.zeros()

        # import pdb; pdb.set_trace()
        # hist_len = (idx-self.price_encode_lens, self.price_encode_lens)
        # date = self.
        
        # Retrieve 
        
        

    def __len__(self):
        return len(self.shifts)

    def maybe_add(self, date, datas, cls_func=dict):
        for data in datas:
            if date not in data:
                data[date] = cls_func()

if __name__ == '__main__':
    import yaml
    import tqdm

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    dataset = StockNetDataset(config, date_range=['2014-07-01','2016-01-01'])
    for i in tqdm.tqdm(range(len(dataset))):
        dataset[i]
        # dataset[i]