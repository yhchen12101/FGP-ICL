# Class-incremental Learning with Rectified Feature-Graph Preservation (ACCV 2020 Oral)

A pytorch implementation of [Class-incremental Learning with Rectified Feature-Graph Preservation, ACCV 2020](https://arxiv.org/abs/2012.08129).

Cheng-Hsun Lei*, Yi-Hsin Chen*, Wen-Hsiao Peng, Wei-Chen Chiu (contributed equally)

![image](https://github.com/yhchen12101/FGP-ICL/blob/master/images/graph(b).png)

## Environment
python 3.7.2

## Example Scripts
1. Cifar100 train from scratch with memory 2000
```
python Rectified_Feature-Graph_Preservation 
```
2. Cifar100 train from 50 with memory 1000
```
python Rectified_Feature-Graph_Preservation -initial=10 -memory=1000
```

## Result
![image](https://github.com/yhchen12101/FGP-ICL/blob/master/images/resutls_table.png)

## Citation
Please consider to cite this paper in your publications if it helps your research:
```
@InProceedings{Lei_2020_ACCV,
    author    = {Lei, Cheng-Hsun and Chen, Yi-Hsin and Peng, Wen-Hsiao and Chiu, Wei-Chen},
    title     = {Class-incremental Learning with Rectified Feature-Graph Preservation},
    booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
    month     = {November},
    year      = {2020}
}
```
