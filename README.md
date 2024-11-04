# ClustersDistillation

_InverseTime: A self-supervised technique for semi-supervised classification of time series_

## Graphical Abstract
![InverseTime](data/InverseTime-Página-1.drawio (3).png)
InverseTime: First we have the dataset with some labeled data. Then, the dataset is transformed by inverting all the series and assigning the pseudo-label 1 to the series in the original order and 0 to the inverted ones. Finally, a convolutional network layer is trained to solve the two tasks.

## Runing Example

```
 python inverse_time.py -p CricketY 0.8 4l
```
Where 0.8 is the unlabel porcentage and CricketY is dataset

## Authors ✒️


* **Manuel Alejandro Goyo**
* **Ricardo Ñanculef**
* **Carlos Valle** 
