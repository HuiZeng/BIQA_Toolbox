# Introduction

This Toolbox contains the source code for the following article:

    @article{DBLP:journals/corr/abs-1708-08190,
      author    = {Hui Zeng and
                   Lei Zhang and
                   Alan C. Bovik},
      title     = {A Probabilistic Quality Representation Approach to Deep Blind Image
                   Quality Prediction},
      url       = {http://arxiv.org/abs/1708.08190},
    }

This Toolbox also aims to provide relatively fair benchmark performances (using the same training and testing splits) of several popular CNN architectures and some classical blind image quality assessment (BIQA) methods using hand-crafted features on four representative image quality assessment (IQA) datasets.


#### Main functions

1. `training_testing_CNNs.m` trains a CNN model using a proportion (e.g. 0.8) of images and tests performance on the remaining images.

2. `crossDatasetTrainTest.m` trains a CNN model on one dataset and tests performance on other datasets.

3. `evaluating_existing_methods.m` evaluates several representative classical BIQA methods including DIIVINE,CORNIA,BRISQUE, NIQE, IL-NIQE, HOSA and FRIQUEE. Except for NIQE, the source codes of other methods need to be downloaded and extracted into the ``supported_methods`` before evaluating these methods.

**How to run the Code**

1. Download the [MatConvNet](http://www.vlfeat.org/matconvnet/) into ``tools`` and Compile it according to the guidence therein. 

2. Create a new fold ``pretrained_models`` and download the pre-trained [AlexNet](http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat) or [ResNet](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) into ``pretrained_models`` if necessary.

3. Create a new fold ``databases``. Download an IQA dataset into ``databases`` and extract the files. Currently, four datasets are supported including the LIVE Challenge, LIVE IQA, CSIQ and TID2013. 

4. Run any of the three main functions.

# Results
The median SRCC(std SRCC) of different methods on different datasets.

|  Methods | LIVE Challenge  | LIVE IQA | CSIQ |  TID2013 |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| AlexNet_SQR | 0.7658 (0.0166)   | 0.9319  |   0.7965   |  0.5362 |
| AlexNet_PQR | 0.8075 (0.0123)  | 0.9554  |   0.8713   |  0.5742 |
| ResNet50_SQR | 0.8236   | 0.9468  | 0.8217 |  0.6406 |
| ResNet50_PQR |  **0.8568**   | **0.9653**  | 0.8728 |  **0.7399** |
| S_CNN_SQR |  0.6582   | 0.9450  | 0.8787 |    0.6526   |
| S_CNN_PQR |  0.6766   | 0.9637  | **0.9080** |    0.6921  |

#### License

This toolbox is made available for research purpose only. 

We utilize the MatConvNet and libSVM toolboxes and re-implement some existing methods. Please check corresponding licence files for details.
