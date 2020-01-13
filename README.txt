
All the six CNN Models (alexnet, googlenet, VGG11, resnet50, resnet152 and inception_v3) are supported by TorchVision. 

cnn-benchmarks have been taken from the below repository (convnet-benchmarks/pytorch):
$git clone https://github.com/mingfeima/convnet-benchmarks.git
$cd convnet-benchmarks/pytorch

Overfeat model was manually ported from Tensorflow to PyTorch, validated and benchmarked. 

Inception_v4 pre-trained model has been taken from the below URL:
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py

cnn-benchmarks now support all the CNN models including ported overfeat model and inceptionv4 taken from above link.

These are the changes made to include support for overfeat and inceptionv4 models:
--------------------------------------------
Place inceptionv4.py and overfeat.py files in the pytorch directory under convnet-benchmarks and add these lines in benchmark.py for batchsize=1:

archs['googlenet'] = [1, 3, 224, 224]
archs['resnet152'] = [1, 3, 224, 224]
archs['overfeat'] = [1, 3, 231, 231]
archs['inceptionv4'] = [1, 3, 299, 299]
--------------------------------------------

There are 5 variants of benchmark.py files for varying batch sizes - 1,32,64,128 and 256 (for example: benchmark-BS1.py for batch size=1 and so on).
