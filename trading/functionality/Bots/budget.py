import numpy as np
from ..metrics_portfolio import BacktestUtility, get_max_drawdown_fast


class Budget:
    def __init__(self, dataset, starting_budget=100, longshort=1,
                period={'days': 1, 'minutes': 0}):
        """Class representing budget over time

        Args:
            dataset: Dataset
            starting_budget: starting point, our initial fund
            longshort: state of bot long or short position
            period: period after which portfolio rebalanced
        """
        self.dataset = dataset
        self.starting_budget = starting_budget
        self.period = period
        self.budget = np.array([starting_budget])
        self.longshort = longshort
        self.budget_short = np.array([starting_budget])
        self.MC = False
    
    def initialize_metrics_class(self):
        """Construct BacktestUtility class with our timestamps"""
        stamp = (self.dataset.manual_end_datetime.timestamp() - self.dataset.manual_start_datetime.timestamp()) / \
                    (self.dataset.manual_end - self.dataset.manual_start)
        time_arr = [self.dataset.manual_start_datetime.timestamp() + stamp*i for i in range(len(self.budget))]
        if not self.MC:
            self.MC = BacktestUtility((time_arr, self.budget))

    def continious_to_budget(self, index_arr):
        """We should input index_arr with the same length
           as continious_budget."""
        assert len(self.budget_short) == len(index_arr)
        arr = [[self.budget_short[i]]*(index_arr[i+1]-index_arr[i]) 
                       for i in range(len(index_arr)-1)]
        return np.array([item for row in arr for item in row])
    
    def get_budget_metrics(self) -> object:
        """"We use continious_budget for calculation speed"""
        drawdown_relative = get_max_drawdown_fast(self.budget, relative=True)[0]
        drawdown_abs = get_max_drawdown_fast(self.budget, relative=False)[0]
        self.MC2 = BacktestUtility(self.budget_short)
        result = self.MC2.get_summary()
        result['max_drawdown_rel'] = drawdown_relative
        result['max_drawdown_abs'] = drawdown_abs
        result = [(i[0], round(i[1], 2)) if type(i[1]) == np.float64 else i for i in list(result.items())]
        return result

    def make_grid(self):
        """"calculating indexes of dates for candels"""
        self._grid_size = int(self.period / self.dataset.interval)
        n_cells = (self.dataset.manual_end_datetime - self.dataset.manual_start_datetime) / self.period
        self._grid = [(self.dataset.manual_start + i*self._grid_size, 
                       self.dataset.manual_start + (i+1)*self._grid_size) for i in range(int(n_cells))] # arrae of index pairs
        assert (self.period / self.dataset.interval - int(self._grid_size)) < 1e-6, "Period should be aliquot to interval"
