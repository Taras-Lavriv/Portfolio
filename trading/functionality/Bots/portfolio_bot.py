import numpy as np
import datetime
import operator
from tqdm.notebook import tqdm
from .budget import Budget


class PortfolioStrategie(Budget):

    def __init__(self, dataset, starting_budget=100, longshort=1, period={'days': 1, 'minutes': 0},
                 filt=False, manual_portfolio=True,
                 n_coins_to_choose=5, n_coins_to_store=3,
                 percentage_down=20):
        """ Initializes Strategie.

        Args:
            dataset: dataset we use for trading
            starting_budget: starting budget for trading
            longshort: state of bot long or short position
            period: period after which portfolio rebalanced
            filt (bool): Indicates if we use a filter.
            manual_portfolio (bool): Indicates if we use a manual portfolio
            n_coins_to_choose: Total coins we choose for trading.
            n_coins_to_store: Number of most profitable coins we stare for the next period.
            percentage_down: Filter criteria.
        """

        super().__init__(dataset, starting_budget, longshort)
        self.period = datetime.timedelta(**period)
        self.filt = filt
        self.portfolio = {}
        self.manual_portfolio = manual_portfolio
        self.n_coins_to_choose = n_coins_to_choose
        self.n_coins_to_store = n_coins_to_store
        self.percentage_down = percentage_down * 0.01
        self.statistic = []
        super().make_grid()

    def statistics_collector(self, index_start, index_end):
        """Collect statistic"""
        current_portf = self.portfolio.copy()
        current_profit = self.dataset.get_profit(index_start, index_end)
        for i in current_portf:
            current_portf[i] = current_profit[i]
        self.statistic.append(current_portf)

    @staticmethod
    def partion_in_decrease(profit):
        """We do not take into acount zero-profitable coins"""
        profit = np.array(profit)
        return np.sum(profit[profit != 1] < 1)/len(profit[profit != 1])

    def portfolio_initialize(self, coins=None, weights=None):
        """Check if terms for manual portfolio are correct"""
        if self.manual_portfolio:
            assert sum(weights) - 1 < 1e-8, "Sum of weights isn't 1 !!!"
            self.portfolio = dict(zip(coins, weights))

    def _choose_best_coins(self, index_start: int, index_end: int):
        """Return dict of the most profitable coins by the period"""
        coins_profit = self.dataset.get_profit(index_start, index_end)
        return sorted(coins_profit.items(), key=operator.itemgetter(1), reverse=True)

    def portfolio_manual_strategie(self, index_start, index_end):
        coins, profits = zip(*self._choose_best_coins(index_start, index_end))
        indices_coins_stored = [i for i, e in enumerate(coins) if e in self.coins_we_trade][:self.n_coins_to_store]
        coins_stored = np.array(coins)[indices_coins_stored].tolist()
        coins_new = np.delete(np.array(coins),
                              [indices_coins_stored])[:(self.n_coins_to_choose-self.n_coins_to_store)].tolist()
        coins = coins_stored + coins_new
        weights = [1/self.n_coins_to_choose for i in range(self.n_coins_to_choose)]
        self.portfolio_initialize(coins, weights)
        self.coins_we_trade = set(self.portfolio.keys())

    def initialize_budget(self):
        self.budg = self.starting_budget
        self.budget_list = [self.starting_budget] * self._grid_size
        self.coins_we_trade, _ = zip(*self._choose_best_coins(*self._grid[0])[:self.n_coins_to_choose])
        self.coins_we_trade = set(self.coins_we_trade)

    def main_with_filter(self):
        """Trading strategy if filter is applied"""
        self.initialize_budget()
        self.coins_we_trade, profit = zip(*self._choose_best_coins(*self._grid[0])[:self.n_coins_to_choose])

        if self.partion_in_decrease(profit) > self.percentage_down:
            self._position = 0
        else:
            self._position = 1
            self.portfolio_manual_strategie(*self._grid[0])

        for grid_start, grid_end in tqdm(self._grid[1:]):
            if self._position == 0:
                self.budget_list.extend([self.budg]*(self._grid_size))
            else:
                period_relative_budget = self.budg * sum(self.dataset._data[coin][grid_start:grid_end]*
                                                         self.portfolio[coin]/self.dataset._data[coin][grid_start]
                                        for coin in self.portfolio.keys())
                self.budget_list.extend(period_relative_budget)
                self.budg = period_relative_budget[-1]

            profit = np.array(list(self.dataset.get_profit(grid_start, grid_end).values()))

            if self.partion_in_decrease(profit) > self.percentage_down:
                self._position = 0
            else:
                self._position = 1
                self.portfolio_manual_strategie(grid_start, grid_end)
            self.statistics_collector(grid_start, grid_end)
        return np.array(self.budget_list)

    def main_no_filter(self):
        """Trading strategy if filter is not applied, first period we don't trade, use uniform weights"""

        self.initialize_budget()
        self.portfolio_manual_strategie(*self._grid[0])

        for grid_start, grid_end in tqdm(self._grid[1:]):
            period_relative_price = self.budg * sum(self.dataset._data[coin][grid_start:grid_end]*self.portfolio[coin]
                                                    /self.dataset._data[coin][grid_start]
                                        for coin in self.portfolio.keys())

            self.budget_list.extend(period_relative_price.tolist())
            self.budg = period_relative_price[-1]
            self.portfolio_manual_strategie(grid_start, grid_end)

            self.statistics_collector(grid_start, grid_end)
        return np.array(self.budget_list)

    def main(self):
        if self.filt == False:
            self.budget = self.main_no_filter()
        else:
            self.budget = self.main_with_filter()
        self.budget_short = self.budget[::self._grid_size]
