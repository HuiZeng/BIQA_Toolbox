# Introduction

This Toolbox contains the source code of the following technical report:

    @article{DBLP:journals/corr/abs-1708-08190,
      author    = {Hui Zeng and
                   Lei Zhang and
                   Alan C. Bovik},
      title     = {A Probabilistic Quality Representation Approach to Deep Blind Image Quality Prediction},
      url       = {http://arxiv.org/abs/1708.08190},
    }

A simplified version of this article was accepted by ICIP 2018. You may cite the accepted paper by:

    @inproceedings{Zeng_PQR,
      author    = {Hui Zeng and
                   Lei Zhang and
                   Alan C. Bovik},
      title     = {BLIND IMAGE QUALITY ASSESSMENT WITH A PROBABILISTIC QUALITY REPRESENTATION},
      booktitle = {2018 IEEE International Conference on Image Processing (ICIP)},
      pages     = {--},
      year      = {2018},
    }

Note that on some datasets such as TID2013, the performance varies greatly when using different training and testing splits (see result tables for details). However, it is too cumbersome for deep BIQA methods to repeat experiments as many times (usually more than 100) as the classical methods. This can make it hard to conduct fair comparisons of results reported in different articles. Thus, this Toolbox also aims to provide relatively fair benchmark performances (by using the same 10 randomly generated training and testing splits, see `/tools/setupDataset/generateTrainingSet.m`) of several popular CNN architectures and some classical blind image quality assessment (BIQA) methods using hand-crafted features on four representative image quality assessment (IQA) datasets.



#### Main functions

1. `training_testing_CNNs.m` trains a CNN model using a proportion (e.g. 0.8) of images and tests performance on the remaining images.

2. `crossDatasetTrainTest.m` trains a CNN model on one dataset and tests performance on other datasets.

3. `evaluating_existing_methods.m` evaluates several representative classical BIQA methods including DIIVINE,CORNIA,BRISQUE, NIQE, IL-NIQE, HOSA and FRIQUEE. Except for NIQE, the source codes of other methods need to be downloaded and extracted into the ``supported_methods`` before evaluating these methods. For convenience, the source codes of these motheds are packed and can be downloaded from [DropBox](https://www.dropbox.com/s/yee4xroe3i4na45/support_methods.zip?dl=0) or [BaiDuYun](https://pan.baidu.com/s/1gfo2Rr9).

**How to run the Code**

1. Download the [MatConvNet](http://www.vlfeat.org/matconvnet/) into ``tools`` and Compile it according to the guidence therein. 

2. Create a new folder ``pretrained_models`` and download the pre-trained [AlexNet](http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat) or [ResNet50](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) into ``pretrained_models`` if necessary.

3. Create a new folder ``databases``. Download an IQA dataset into ``databases`` and extract the files. Currently, four datasets are supported including the [LIVE Challenge](http://live.ece.utexas.edu/research/ChallengeDB/index.html), [LIVE IQA](http://live.ece.utexas.edu/research/quality/subjective.htm), [CSIQ](http://vision.eng.shizuoka.ac.jp/mod/page/view.php?id=23) and [TID2013](http://www.ponomarenko.info/tid2013.htm). 

4. The 64-bit compiled libSVM are already included in `tools`. If they are not compatible to your device, please download the [source code](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) and compile it on your own device.

5. Run any of the three main functions.

# Results
The median SRCC (std SRCC) of 10 repititions of different methods on different datasets. Note that the results of CNN models may be slightly different because of the randomness in the training process of CNN models.

|    Methods   | LIVE Challenge        | LIVE IQA             | CSIQ                |  TID2013             |
|:------------:|:---------------------:|:--------------------:|:-------------------:|:--------------------:|
| DIIVINE      | 0.5761 (0.0299)       | 0.8787 (0.0554)      | 0.7835 (0.1643)     |  0.5829 (0.0605)     |
| CORNIA       | 0.6323 (0.0360)       | 0.9420 (0.0118)      | 0.7299 (0.0338)     |  0.6226 (0.0760)     |
| BRISQUE      | 0.6085 (0.0284)       | 0.9374 (0.0260)      | 0.7502 (0.0489)     |  0.5258 (0.0566)     |
| NIQE         | 0.4347 (0.0286)       | 0.9154 (0.0133)      | 0.6298 (0.0284)     |  0.2992 (0.0412)     |
| IL-NIQE      | 0.4254 (0.0286)       | 0.9017 (0.0278)      | 0.8066 (0.0252)     |  0.5185 (0.0469)     |
| HOSA         | 0.6610 (0.0440)       | 0.9477 (0.0080)      | 0.7812 (0.0729)     |  0.6876 (0.0925)     |
| FRIQUEE-ALL  | 0.6873 (0.0253)       | 0.9507 (0.0275)      | 0.8414 (0.1238)     |  0.7133 (0.0703)     |
| AlexNet_SQR  | 0.7658 (0.0166)       | 0.9319 (0.0269)      | 0.7965 (0.0394)     |  0.5362 (0.1100)     |
| AlexNet_PQR  | 0.8075 (0.0123)       | 0.9554 (0.0185)      | 0.8713 (0.0229)     |  0.5742 (0.1156)     |
| ResNet50_SQR | 0.8236 (0.0159)       | 0.9468 (0.0199)      | 0.8217 (0.0481)     |  0.6406 (0.0577)     |
| ResNet50_PQR |  **0.8568 (0.0095)**  | **0.9653 (0.0105)**  | 0.8728 (0.0319)     |  **0.7399 (0.1037)** |
| S_CNN_SQR    |  0.6582 (0.0323)      | 0.9450 (0.0320)      | 0.8787 (0.0213)     |  0.6526 (0.1136)     |
| S_CNN_PQR    |  0.6766 (0.0326)      | 0.9637 (0.0223)      | **0.9080 (0.0212)** |  0.6921 (0.1246)     |

The median PLCC (std PLCC) of 10 repititions of different methods on different datasets.

|    Methods   |    LIVE Challenge     |    LIVE IQA         |         CSIQ        |       TID2013        |
|:------------:|:---------------------:|:-------------------:|:-------------------:|:--------------------:|
| DIIVINE      | 0.5955 (0.0303)       | 0.8813 (0.0379)     | 0.8362 (0.1710)     |  0.6723 (0.0641)     |
| CORNIA       | 0.6613 (0.0356)       | 0.9457 (0.0119)     | 0.8036 (0.0253)     |  0.7038 (0.0669)     |
| BRISQUE      | 0.6465 (0.0361)       | 0.9448 (0.0274)     | 0.8286 (0.0380)     |  0.6331 (0.0568)     |
| NIQE         | 0.4784 (0.0264)       | 0.9194 (0.0112)     | 0.7181 (0.0293)     |  0.4154 (0.0567)     |
| IL-NIQE      | 0.5066 (0.0247)       | 0.8654 (0.0866)     | 0.8083 (0.0726)     |  0.6398 (0.0867)     |
| HOSA         | 0.6750 (0.0299)       | 0.9492 (0.0089)     | 0.8415 (0.0522)     |  0.7637 (0.0654)     |
| FRIQUEE-ALL  | 0.7096 (0.0322)       | 0.9576 (0.0264)     | 0.8733 (0.1209)     |  0.7755 (0.0716)     |
| AlexNet_SQR  | 0.8074 (0.0176)       | 0.9426 (0.0227)     | 0.8405 (0.0384)     |  0.6136 (0.1045)     |
| AlexNet_PQR  | 0.8357 (0.0124)       | 0.9638 (0.0181)     | 0.8958 (0.0219)     |  0.6687 (0.0840)     |
| ResNet50_SQR | 0.8680 (0.0117)       | 0.9527 (0.0145)     | 0.8713 (0.0355)     |  0.7068 (0.0695)     |
| ResNet50_PQR | **0.8822 (0.0098)**   | **0.9714 (0.093)**  | 0.9010 (0.0266)     |  **0.7980 (0.0848)** |
| S_CNN_SQR    | 0.6729 (0.0309)       | 0.9455 (0.0249)     | 0.8987 (0.0234)     |  0.6921 (0.1314)     |
| S_CNN_PQR    | 0.7032 (0.0298)       | 0.9656 (0.0219)     | **0.9267 (0.0219)** |  0.7497 (0.1089)     |

# License

This toolbox is made available for research purpose only. 

We utilize the MatConvNet and libSVM toolboxes and re-implement some existing methods. Please check their licence files for details.
