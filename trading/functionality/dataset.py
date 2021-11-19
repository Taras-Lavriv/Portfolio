import requests
import numpy as np
import os
import re
from datetime import datetime, timedelta


class Dataset:
    """Load dataset either from binance, or from file."""

    url = 'https://api.binance.com/api/v3/klines?symbol={}&interval={}&startTime={}&endTime={}&limit=1000'
    data_dir = 'data/'

    def __init__(self, interval={'minutes': 1}, data_source='folder'):
        """Initializes dataset.

        Args:
            data_source: Source to load the data.
            interval: Time interval we use.
        """
        self.data_source = data_source
        self.interval = timedelta(**interval)

    @classmethod
    def get_from_binance(cls, startpoint, endpoint, market='BTCUSDT', tick_interval='1m'):
        """Retrieve market data from binance.

        Args:
            startpoint: start time
            endpoint: end time
            market: trading pair
            tick_interval: interval between trades
        Returns:
            raw data
        """
        url1 = cls.url.format(market, tick_interval,
                              int(startpoint.timestamp()) * 1000,
                              int(endpoint.timestamp()) * 1000)
        return requests.get(url1).json()

    @classmethod
    def make_start_price_list(cls, start, finish, pair):
        """Make list of prices.

        Args:
            start: start time
            finish: end time
            pair: pair
        Returns:
            list of prices
        """
        data = []
        while start < finish:
            line = cls.get_from_binance(start, finish, pair)
            for i in line:
                data.append(int(float(i[1])))
            start += timedelta(minutes=1000)
        return data

    @classmethod
    def create_data_file(cls, start, finish, pair):
        """Create np file with list of prices.

        Args:
            start: start time
            finish: end time
            pair: pair
        """
        np.save('data/{}_{}_{}.npy'.format(pair, start, finish),
                cls.make_start_price_list(datetime.strptime(start, '%Y-%m-%d'),
                                          datetime.strptime(finish, '%Y-%m-%d'), pair))

    @classmethod
    def get_from_file(cls):
        """Download trading data from file.

        Returns:
            data: dict of prices for pair
            time_start: start time of trading for pair
            time_end: end time of trading for pair
        """
        data, time_start, time_end = {}, {}, {}
        for file in os.listdir(cls.data_dir):
            if file.endswith(".npy"):
                pair, start, end, _ = re.split('_|\.', file)
                data[pair] = np.load(cls.data_dir + file)
                time_start[pair] = datetime.strptime(start, '%Y-%m-%d').timestamp()
                time_end[pair] = datetime.strptime(end, '%Y-%m-%d').timestamp()
        return data, time_start, time_end

    @classmethod
    def update_data_file(cls):
        """Update data files to the current."""
        for file in os.listdir(cls.data_dir):
            if file.endswith(".npy"):
                pair, start, end, _ = re.split('_|\.', file)
                new_end = datetime.now()
                new_end_str = new_end.strftime('%Y-%m-%d')
                if end != new_end_str:
                    old_data = np.load(cls.data_dir + file)
                    new_data = cls.make_start_price_list(datetime.strptime(end, '%Y-%m-%d'), new_end, pair)
                    updated_data = np.hstack((old_data, new_data))
                    np.save('{}/{}_{}_{}.npy'.format(cls.data_dir, pair, start, new_end_str), updated_data)
                    os.remove((cls.data_dir + '/' + file))

    def _get_data(self):
        """Initialises data"""
        if self.data_source == 'folder':
            self._data, self._time_start, self._time_end = Dataset.get_from_file()
            self._longest = self._key_longest_data_array()
        assert self.interval.seconds % 60 == 0
        self._data = {key: value[::self.interval.seconds // 60] for key, value in self._data.items()}

    def _key_longest_data_array(self):
        """Returns key of the longest data array."""
        length = 0
        longest = ''
        for i in self._data:
            if len(self._data[i]) > length:
                length = len(self._data[i])
                longest = i
        return longest

    def _expanding_data_array(self):
        """Expanding all data arrays to the length of the longest one."""
        for i in self._data:
            if len(self._data[i]) < len(self._data[self._longest]):
                a = [self._data[i][0] for j in range(len(self._data[self._longest])-len(self._data[i]))]
                a.extend(self._data[i])
                self._data[i] = np.array(a)

    def earliest_latest(self):
        """Initialises time"""
        self._earliest = self._time_start[self._longest]
        self._latest = self._time_end[self._longest]
        
    def index_to_date(self, index):
        """Returns datetime object by data array index"""
        tick_interv = (self._time_end[self._longest] - self._time_start[self._longest]) / len(self._data[self._longest])
        indexes_date = self._time_start[self._longest] + index * tick_interv
        return indexes_date

    def date_to_index(self, date):
        """Returns closest index by datetime"""
        tick_interv = (self._time_end[self._longest] - self._time_start[self._longest]) / len(self._data[self._longest])
        dates_index = int((date.timestamp() - self._time_start[self._longest]) / tick_interv)
        return dates_index
    
    def index_to_str_date(self, index: int):
        """Returns date in string format"""
        return datetime.fromtimestamp(self.index_to_date(index)).strftime('%Y-%m-%d')

    def slice_data(self, start: str, end: str):
        """
        Args:
            start: start of slice.
            end: end of slice.
        Returns: slice of data.
        """
        if datetime.strptime(start, '%Y-%m-%d') < datetime.fromtimestamp(self._time_start['BTCUSDT']):
            raise ValueError('Data earlier than {} is not available'
                             .format(datetime.fromtimestamp(self._time_start['BTCUSDT'])))
        
        self.manual_start_datetime = datetime.strptime(start, '%Y-%m-%d')
        self.manual_end_datetime = datetime.strptime(end, '%Y-%m-%d')

        self.manual_start = self.date_to_index(datetime.strptime(start, '%Y-%m-%d'))
        self.manual_end = self.date_to_index(datetime.strptime(end, '%Y-%m-%d'))

    def __len__(self):
        """Returns len of data"""
        return len(self._data)
            
    def get_trading_pairs(self):
        """Returns trading pairs"""
        return self._data.keys()
    
    def get_profit(self, index_start: int, index_end: int):
        """
            Args:
                index_start: first element.
                index_end: last element
            Returns:
                profit for pair
        """
        return {coin: (self._data[coin][index_end]/self._data[coin][index_start])
                for coin in self._data.keys()}
