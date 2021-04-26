This repository contains CNN and RNN benchmarks for PyTorch.

All the six CNN Models (alexnet, googlenet, VGG11, resnet50, resnet152 and inception_v3) are supported by TorchVision. 

cnn-benchmarks have been taken from the below repository (convnet-benchmarks/pytorch):
$git clone https://github.com/mingfeima/convnet-benchmarks.git
$cd convnet-benchmarks/pytorch

Overfeat model was ported from Tensorflow to PyTorch, validated and benchmarked. 

Inceptionv4 pre-trained model has been taken from the below URL:
https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/inceptionv4.py

cnn-benchmarks now support all the six CNN models mentioned above including ported overfeat model and inceptionv4 taken from above link.

These are the changes made to include support for overfeat and inceptionv4 models:
--------------------------------------------
Place inceptionv4.py and overfeat.py files in the pytorch directory under convnet-benchmarks and add these lines in benchmark.py for batchsize=1:

archs['googlenet'] = [1, 3, 224, 224]
archs['resnet152'] = [1, 3, 224, 224]
archs['overfeat'] = [1, 3, 231, 231]
archs['inceptionv4'] = [1, 3, 299, 299]
--------------------------------------------

To try out CNN benchmarks for varying batch sizes, we need to change batchsize in benchmark.py as below:
archs['googlenet'] = [1, 3, 224, 224] for Batch Size=1
archs['googlenet'] = [128, 3, 224, 224] for Batch Size=128
Sample Run : export OMP_NUM_THREADS=64 && numactl --cpunodebind=0-3 --interleave=0-3 python benchmark.py --arch=alexnet

Optimized benchmark code with support for varying batch sizes : benchmark-optimized.py
Sample Run : export OMP_NUM_THREADS=64 && numactl --cpunodebind=0-3 --interleave=0-3 python benchmark-optimized.py --arch=alexnet --batch_size=128

Benchmark code with PyTorch Profiler Enabled : benchmark-pytorch-profiler.py
Sample Run : export OMP_NUM_THREADS=64 && numactl --cpunodebind=0-3 --interleave=0-3 python benchmark-optimized.py --arch=alexnet --batch_size=128

Validation Code which checks the outputs against DNNL output for varying OMP_NUM_THREADS and batch sizes: benchmark-validation.py
Code to generate reference DNNL outputs : cnn_pytorch_output_gen_mkl.sh
Sample Run : numactl --cpunodebind=0-3 --interleave=0-3 python benchmark-validation.py --arch=alexnet

There are three RNN Benchmarks (RNN, LSTM and Basic_LSTM) which have been ported from Tensorflow.
(Tensorflow RNN benchmarks are part of coral2 deep learning suite).
One can specify the network type, hidden layer size, input size, sequence length and batch size as shown below:
Sample Run : export OMP_NUM_THREADS=64 && numactl --cpunodebind=0-3 --interleave=0-3 python rnn-pytorch.py -n RNN -l 512 -i 512 -s 256 -b 32
