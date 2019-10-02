# %% md
# Compute rank order statistics on annotated frame importance scores
import sys

import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata

sys.path.append('/home/emma/summary_evaluation/')
from summ_evaluator import SummaryEvaluator


class RankCorrelationEvaluator(SummaryEvaluator):
    def __init__(self, ds_name, model_name, num_classes):
        super().__init__(ds_name=ds_name, model_name=model_name, num_classes=num_classes)

        self.corr = None
        self.p_value = None

    def rc_func(self, x, y, metric):
        if metric == 'kendalltau':
            return kendalltau(rankdata(-x), rankdata(-y))
        elif metric == 'spearmanr':
            return spearmanr(x, y)
        else:
            raise RuntimeError

    def get_metrics(self, metric='kendalltau', random=False, threshold=None, print_output=False):
        y = self.get_y(random, threshold)
        self.corr, self.p_value = self.rc_func(self.y_true, y, metric)
        if print_output:
            print(self.rc_func(self.y_true, y, metric))
        return self.corr

    def plot(self):
        pass


if __name__ == '__main__':

    metric = 'kendalltau'
    rank_corr_eval = RankCorrelationEvaluator('run_1_two', num_classes=2)
    mean_res = rank_corr_eval.get_metrics(metric)
    print('pred' + ': mean %.3f' % (mean_res))
