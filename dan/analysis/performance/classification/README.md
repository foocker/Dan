## Support
- [x] [TopKAccuracy](performace/classification/accuracy.py)
- [x] [ConfusionMatrix](performace/classification/cfx_based_metrics.py)
- [x] [CMAccuracy](performace/classification/cfx_based_metrics.py)
- [x] [CMPrecisionRecall](performace/classification/cfx_based_metrics.py)
- [x] [Fbetascore](performace/classification/fbeta_score.py)
- [x] [APscore](performace/classification/average_precision_score.py)
- [x] [mAPscore](performace/classification/average_precision_score.py)
- [x] [PRcurve](performace/classification/pr_curve.py)
- [x] [AUCscore](performace/classification/roc_auc_score.py)
- [x] [ROCcurve](performace/classification/roc_curve.py)

## Usage
### Known Issues
- the inputs (pred , target, etc.) of all metrics have to be numpy array.
- the output value of all metrics are stored in a dict format.
 
Take topk accuracy for example in pytorch classification training:

```shell
from dan.analysis.performance.classification import TopKAccuracy

for epoch in range(num_max_epoch):
    topkacc = TopKAccuracy(topk=(1, 3, 5))
    for (image, target) in dataloader:
        ...
        pred = model(image)
        loss = ...
        ...
        # calculate acc for current batch
        acc_current_batch = topkacc(pred.numpy(), target.numpy())
        # calculate average acc from the start of current epoch till current batch
        acc_current_average = topkacc.accumulate()
    # calculate acc of the epoch
    acc_epoch = topkacc.accumulate()

>>> acc_epoch
{'top_1': 0.6, 'top_3': 0.8, 'top_5': 0.9}

```
