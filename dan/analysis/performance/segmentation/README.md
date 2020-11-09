## Support
- [x] [ConfusionMatrix](performace/segmentation/cfx_based_metrics.py)
- [x] [Accuracy](performace/segmentation/cfx_based_metrics.py)
- [x] [IoU](performace/segmentation/cfx_based_metrics.py)
- [x] [mIoU](performace/segmentation/cfx_based_metrics.py)
- [x] [DiceScore](performace/segmentation/cfx_based_metrics.py)

## Usage
### Known Issues
- the inputs (pred , target, etc.) of all metrics have to be numpy array.
- the output value of all metrics are stored in a dict format.
 
Take miou for example in pytorch segmentation training:

```shell
from dan.analysis.performance.segmentation import mIoU

for epoch in range(num_max_epoch):
    miou = mIoU(num_classes=10)
    for (image, target) in dataloader:
        ...
        pred = model(image)
        loss = ...
        ...
        # calculate miou for current batch
        miou_current_batch = miou(pred.numpy(), target.numpy())
        # calculate miou from the start of current epoch till current batch
        miou_current_average = miou.accumulate()
    # calculate miou of the epoch
    miou_epoch = miou.accumulate()

>>> miou_epoch
{'mIoU': 0.8}

```
