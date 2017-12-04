# Introducetion
This Toolbox aims to benchmark the performance of several popular CNN architectures on four representative image quality assessment (IQA) datasets.


#### Installation


**How to run the Code**

1. Download the MatConvNet into ``tools`` and Compile it according to the guidence on [website](http://www.vlfeat.org/matconvnet/)

2. Create a new fold ``pretrained_models`` and download the pre-trained [AlexNet](http://www.vlfeat.org/matconvnet/models/imagenet-caffe-alex.mat) or [ResNet](http://www.vlfeat.org/matconvnet/models/imagenet-resnet-50-dag.mat) into ``pretrained_models``.

3. Create a new fold ``databases``. Download an IQA dataset into ``databases`` and extract the files. Currently, four datasets are supported including the LIVE Challenge, LIVE IQA, CSIQ and TID2013.
