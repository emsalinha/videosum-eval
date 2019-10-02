from phash_evaluator import PhashEvaluator
from class_evaluator import ClassificationEvaluator
from rank_corr_evaluator import RankCorrelationEvaluator
import matplotlib.pyplot as plt
import numpy as np
save = False
ds_names = ['MovieSum', 'TVsum', 'SumMe']

np.set_printoptions(precision=2)

# ds_name = 'SumMe'
# model_name = '{}_one'.format(ds_name.lower())
# n_classes = 1
# class_eval = ClassificationEvaluator(ds_name=ds_name.lower(), model_name=model_name, num_classes=n_classes)
# f1_score = class_eval.get_metrics(print_output=True)
# f1_score = class_eval.get_metrics(random=True, print_output=True)
# class_eval.plot(normalize=False, title='Confusion matrix, {} dataset'.format(ds_name))
#
# ds_name = 'MovieSum'
# model_name = '{}_two'.format(ds_name.lower())
# n_classes = 2
# class_eval = ClassificationEvaluator(ds_name=ds_name.lower(), model_name=model_name, num_classes=n_classes)
# f1_score = class_eval.get_metrics(print_output=False)
# print(f1_score)
# f1_score = class_eval.get_metrics(random=True, print_output=False)
# print(f1_score)
# class_eval.plot(normalize=False, title='Confusion matrix, {} dataset'.format(ds_name))

ds_name = 'MovieSum'
model_name = '{}_two'.format(ds_name.lower())
n_classes = 2
class_eval = ClassificationEvaluator(ds_name=ds_name.lower(), model_name=model_name, num_classes=n_classes)
f1_score = class_eval.get_metrics(print_output=True)
print(f1_score)
f1_score = class_eval.get_metrics(random=True, print_output=True)
print(f1_score)
class_eval.plot(normalize=False, title='Confusion matrix, {} dataset'.format(ds_name))
#

metric = 'kendalltau'
rank_eval = RankCorrelationEvaluator(ds_name=ds_name.lower(),model_name=model_name, num_classes=n_classes)
rank_score = rank_eval.get_metrics(metric=metric, print_output=True)
rank_score_random = rank_eval.get_metrics(metric=metric, random=True, print_output=True)


metric = 'hamming'
phash_eval = PhashEvaluator(ds_name=ds_name.lower(),model_name=model_name, num_classes=n_classes)
phash_sim, phash_non_pred_sim = phash_eval.get_metrics(metric=metric, random=False, print_output=False)
print(phash_non_pred_sim-phash_sim)
phash_sim_random, phash_non_pred_sim_random = phash_eval.get_metrics(metric=metric, random=True, print_output=False)
print(phash_non_pred_sim_random-phash_sim_random)
#
# phash_eval.plot()
# phash_eval.plot(random=True)

