from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as md
from datetime import datetime


class BacktestUtility:
    """Utility for providing backtest summary.
    Uses either 2-tuple of budget snapshots and their timestamps or
    array-like budget snapshots.
    """

    def __init__(self, budget_tuple):
        """Initializes

        :param budget_tuple:
        """
        assert not isinstance(budget_tuple, tuple) or len(budget_tuple) == 2
        if not isinstance(budget_tuple, tuple):
            warn('Timestamps are not provided, generating index.')
            self.times = np.array(list(range(len(budget_tuple))))
            self.budget = np.array(budget_tuple)
        else:
            self.times = np.array(budget_tuple[0])
            self.budget = np.array(budget_tuple[1])
        self.deal_ind = np.insert(np.diff(self.budget).astype(np.bool),
                                  0, True)
        self.deal_times = self.times[self.deal_ind]
        self.deals = self.budget[self.deal_ind]

    def get_revenue(self, relative=True):
        """
        Returns total returns for the given budget.
        :param relative: bool. If True - returns relative returns,
                               otherwise returns are absolute.
        :return: float. Relative of absolute returns.
        """
        assert isinstance(relative, bool)
        if relative:
            rev = (self.budget[-1]-self.budget[0]) / self.budget[0]
        else:
            rev = self.budget[-1] - self.budget[0]
        return rev

    def get_max_drawdown(self, relative=True):
        """
        Returns maximum drawdown for the given budget.
        :param relative: bool. If True - returns relative drawdown,
                               otherwise drawdown is absolute.
        :return: (float, int) (Relative of absolute drawdown, 
                               start time of this drawdown).
                 
        """
        assert isinstance(relative, bool)
        drawdown_val = 0
        for i in range(len(self.deals)-1):
            cur_max = self.deals[i]
            cur_min = min(self.deals[i:])
            if relative:
                cur_dd = cur_min / cur_max - 1
            else:
                cur_dd = cur_min - cur_max
            if cur_dd < drawdown_val:
                drawdown_val = cur_dd

        return drawdown_val, i

    def get_sharpe(self):
        """
        Returns Sharpe coefficient for given deals.
        :return: float. Sharpe coefficient value.
        """
        returns = np.diff(self.deals) / self.deals[:-1]
        ret_mean = returns.mean()
        ret_std = returns.std()
        return ret_mean/ret_std

    def get_profit_factor(self):
        """
        Calculates profit factor.
        :return: num. Profit factor.
        """
        ret = np.diff(self.deals)
        return sum(ret[ret > 0]) / abs(sum(ret[ret < 0]))

    def get_deals_summary(self):
        """
        Returns summary stats for deals: number of total/positive/negative
        deals, average positive/negative deal returns.
        :return: OrderedDict. Deal summary with measure names as keys.
        """
        deal_delta = np.diff(self.deals)
        out = {'n_deals': len(deal_delta)}
        deal_rev = deal_delta / self.deals[:-1]
        deals_neg = deal_rev[deal_rev < 0]
        deals_pos = deal_rev[deal_rev > 0]
        out['n_neg'], out['n_pos'] = len(deals_neg), len(deals_pos)
        out['avg_neg'], out['avg_pos'] = deals_neg.mean(), deals_pos.mean()
        return out

    def budget_plot(self, figsize=(10, 7), log_scale=False):
        """
        Creates a simple budget plot.
        Call with BacktestUtility.budget_plot().
        :return: list. matplotlib plot object.
        """
        xfmt = md.DateFormatter('%Y-%m-%d')
        
        dates = [datetime.fromtimestamp(ts) for ts in self.times]
        datenums = md.date2num(dates)
        
        plt.figure(figsize=figsize)
        if log_scale:
            b_plt = plt.plot(datenums, np.log(self.budget) / np.log(self.budget[0]))
        else:
            b_plt = plt.plot(datenums, self.budget)
        
        ax = plt.gca()
        ax.xaxis.set_major_formatter(xfmt)
        
        plt.xticks(rotation=30)
        plt.title('Budget plot')
        plt.xlabel('Time')
        plt.ylabel('Budget, USD')
        plt.show()
        return b_plt

    def get_longest_consecutive(self):
        """
        Returns the maximum length of consecutive positive and negative
        deals.
        :return: dict. {'longest_neg': max(neg_deals),
                        'longest_pos': max(pos_deals)}
        """
        deal_signs = np.sign(np.diff(self.deals))
        cur_sign = deal_signs[0]
        cur_seq_len = 0
        pos_deals = []
        neg_deals = []
        for i in deal_signs:
            if i == cur_sign:
                cur_seq_len += 1
            else:
                if cur_sign == 1:
                    pos_deals.append(cur_seq_len)
                else:
                    neg_deals.append(cur_seq_len)
                cur_seq_len = 1
                cur_sign *= -1
        if cur_sign == 1:
            pos_deals.append(cur_seq_len)
        else:
            neg_deals.append(cur_seq_len)
        return {'longest_neg': max(neg_deals), 'longest_pos': max(pos_deals)}

    def get_summary(self):
        """
        Returns all summary stats.
        :return: OrderedDict. A collection of all summary stats.
        """
        out = {'total_returns': self.get_revenue()}
        deals_summary = self.get_deals_summary()
        for item in deals_summary:
            out[item] = deals_summary[item]
        out['max_drawdown_abs'] = (self.get_max_drawdown(False)[0])
        out['max_drawdown_rel'] = (self.get_max_drawdown()[0])
        consec_info = self.get_longest_consecutive()
        out['longest_neg'] = consec_info['longest_neg']
        out['longest_pos'] = consec_info['longest_pos']
        out['sharpe'] = self.get_sharpe()
        out['profit_factor'] = self.get_profit_factor()
        out['rec_factor'] = -self.get_revenue(False) / out['max_drawdown_abs']
        return out


def get_drawdown_simple(budget, relative=True):
    """
    Returns maximum drawdown for the given budget.
    :param relative: bool. If True - returns relative drawdown,
                           otherwise drawdown is absolute.
    :return: (float, int) (Relative of absolute drawdown, 
                           start time of this drawdown).

    """
    assert isinstance(relative, bool)
    drawdown_val = 0
    drawdown_time = 0
    for i in range(len(budget)-1):
        cur_max = budget[i]
        cur_min = min(budget[i:])
        if relative:
            cur_dd = cur_min / cur_max - 1
        else:
            cur_dd = cur_min - cur_max
        if cur_dd < drawdown_val:
            drawdown_val = cur_dd
            drawdown_time = (i, np.argmin(budget[i:]))

    return drawdown_val, drawdown_time


def get_max_drawdown_fast(budget, size_to_run=1000, alpha=10, relative=True):
    """
    Returns maximum drawdown for the given budget.
    For small budget sizes itt is preferable to use 
    :param size_to_run: on such size we first run get_drawdown_simple
                        to avoid collisions more stable get_drawdown_simple
    :param alpha: made for faster convergence, set 1 for 
    
    :return: (float, (int, int)) (Relative of absolute drawdown, 
                           (start, end) time of this drawdown).
    """
    assert isinstance(relative, bool)
    drawdown_val, drawdown_time = get_drawdown_simple(budget[:size_to_run], relative=relative)
    index_start = size_to_run
    
    while True:
        try:
            am = np.argmin(budget[index_start+alpha:])
        except Exception:
            return drawdown_val, drawdown_time
        
        if am == 0:
            index_start += alpha
            continue
            
        cur_min = budget[index_start + alpha + am]
        cur_max_index = np.argmax(budget[index_start:index_start+am])
        cur_max = np.max(budget[index_start:index_start+am])
        
        if relative:
            cur_dd = cur_min / cur_max - 1
        else:
            cur_dd = cur_min - cur_max

        if cur_dd < drawdown_val:
            drawdown_time = (index_start+cur_max_index, index_start+alpha+am)
            drawdown_val = cur_dd
        
        # move through array
        index_start += (am + alpha)
        for i in range(index_start, len(budget)):
            if budget[i] > cur_max:
                index_start = i
                break

    return drawdown_val, drawdown_time





