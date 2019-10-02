import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from summ_evaluator import SummaryEvaluator

class ClassificationEvaluator(SummaryEvaluator):
    def __init__(self, ds_name, model_name, num_classes):
        super().__init__(ds_name, model_name, num_classes)

        #metrics
        self.report_dict = None

        self.precision = None
        self.recall = None
        self.f1_score = None
        self.support = None

        self.true_negatives = None
        self.false_positives = None
        self.false_negatives = None
        self.true_positives = None

    def get_metrics(self, random=False, threshold=None, print_output=False, colorbar=False):
        y = self.get_y(random, threshold)

        self.report_dict = skm.classification_report(self.y_true, y, output_dict=True)
        self.precision = self.report_dict['1']['precision']
        self.recall = self.report_dict['1']['recall']
        self.f1_score = self.report_dict['1']['f1-score']
        self.support = self.report_dict['1']['support']
        if print_output:
            print(self.report_dict)

        tn, fp, fn, tp = skm.confusion_matrix(self.y_true, y).ravel()
        self.true_negatives = tn
        self.false_positives = fp
        self.false_negatives = fn
        self.true_positives = tp
        if print_output:
            print('tn: {}, fp: {}, fn: {}, tp: {}'.format(tn, fp, fn, tp))

        return skm.f1_score(self.y_true, y)

    def plot(self, normalize=False, title=None, cmap=plt.cm.Blues, random=False, threshold=None, colorbar=False):
        y = self.get_y(random, threshold)
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, {}'.format(self.ds_name)

        # Compute confusion matrix
        cm = confusion_matrix(self.y_true, y)
        # Only use the labels that appear in the data
        self.class_names = ['0', '1']

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, {}'.format(self.ds_name))

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        if colorbar:
            ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=self.class_names, yticklabels=self.class_names,
               ylabel='True label',
               xlabel='Predicted label')

        ax.yaxis.label.set_size('xx-large')
        ax.xaxis.label.set_size('xx-large')
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), ha="center")  # , rotation=45, rotation_mode="anchor")
        for i, t in enumerate(ax.get_xticklabels()):
            t.set_fontsize('xx-large')
            #plt.setp(ax.get_yticklabels(), ha="center", rotation=90, rotation_mode="anchor")
        for i, t in enumerate(ax.get_yticklabels()):
            t.set_fontsize('xx-large')
            t.set_rotation('vertical')
            if i == 0:
                ha = 'right'
                t.set_ha(ha)
            else:
                ha = 'right'
                t.set_ha(ha)

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            if i == 0:
                va = 'top'
                y = i + 0.2
            else:
                va = 'bottom'
                y = i - 0.2
            for j in range(cm.shape[1]):
                ax.text(j, y, format(cm[i, j], fmt),
                        ha="center", va=va, size='xx-large',
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        plt.savefig('/home/emma/summary_evaluation/results/confusion_{}.pdf'.format(self.ds_name))


if __name__ == '__main__':

    class_eval = ClassificationEvaluator('run_1_two', num_classes=2)

    class_eval.get_metrics()

    # Plot non-normalized confusion matrix
    class_eval.plot(title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    class_eval.plot(normalize=True, title='Normalized confusion matrix')

    plt.show()