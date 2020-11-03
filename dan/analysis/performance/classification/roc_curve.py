import os
import time

import numpy as np
import matplotlib.pyplot as plt

from .base import BaseScoreCurve
from .libs import roc_curve


class ROCcurve(BaseScoreCurve):
    """
    Calculate value(true_positive_rate, false_positive_rate, thresholds) and
    export image of roc curve.
    Note: this implementation is restricted to the binary, multiclass classification
          task.

    Args:
        num_classes (int): number of classes. For num_classes=n, default
                           categorical indexes are 0 ~ n-1
    Examples:
        >>> roc_curve = ROCCurve(num_classes=5)
        >>> for batch in dataloader:
        ...     # calculate roc value of current batch
        ...     roc_curve_batch = roc_curve(pred, target)
        >>> # calculate roc value of the epoch
        >>> roc_curve_epoch = roc_curve.accumulate()
        >>> # export image of roc curve
        >>> roc_curve.export(export_path='.', )
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def accumulate(self):
        accumulate_state = {}
        for cat_id in range(self.num_classes):
            fpr, tpr, thresholds = roc_curve(self.y_true,
                                             self.probas_pred[:, cat_id],
                                             pos_label=cat_id)
            accumulate_state[str(cat_id)] = {
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'thresholds': thresholds,
            }
        return accumulate_state

    def export(self, export_path='.', **kwargs):

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        os.makedirs(export_path, exist_ok=True)

        accumulate_state = self.accumulate()

        for cat_id in accumulate_state.keys():

            plt.figure(11, figsize=(9, 9), dpi=400)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.grid(True)
            plt.plot(np.arange(0.0, 1.01, 0.01), np.arange(0.0, 1.01, 0.01),
                     color='royalblue', linestyle='--')
            plt.annotate("random guess", xy=(0.5, 0.5), color='royalblue',
                         rotation=45, xytext=(-20, 0), textcoords='offset points')

            line_kwargs = {
                'label': f'cat_id: {cat_id}',
                'color': 'crimson',
                'linestyle': '-',
                'linewidth': 1,
            }
            line_kwargs.update(**kwargs)

            plt.plot(accumulate_state[cat_id]['false_positive_rate'],
                     accumulate_state[cat_id]['true_positive_rate'],
                     **line_kwargs)

            plt.legend(loc='lower left')
            plt.tight_layout()
            plt.savefig(os.path.join(export_path, timestamp + f'_roc_curve_of_cat_{cat_id}'), dpi=400)
            plt.close()
