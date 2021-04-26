import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import subprocess
from collections import OrderedDict
import torch.autograd.profiler as profiler
from mobilenet import MobileNetV2
models.__dict__['mobilenet_v2'] = MobileNetV2
import os
from shufflenet import ShuffleNet
models.__dict__['shufflenet'] = ShuffleNet

from unet2d import UNet
models.__dict__['unet'] = UNet

from unet3d import UNet3D
models.__dict__['unet3d'] = UNet3D

#from googlenet import GoogLeNet
#models.__dict__['googlenet'] = GoogLeNet

from overfeat import Overfeat
models.__dict__['overfeat'] = Overfeat

from inceptionv4 import InceptionV4
models.__dict__['inceptionv4'] = InceptionV4



def benchmark(args, archs_list, steps, nDryRuns):
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    arch_dict = {args.arch: archs[args.arch]} if args.arch in archs_list else archs # by huiming, support one or all models.

    if args.cuda:
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        cudnn.deterministic = True

        kernel = 'cudnn'
        p = subprocess.check_output('nvidia-smi --query-gpu=name --format=csv', 
                                    shell=True)
        device_name = str(p).split('\\n')[1]
    else:
        kernel = 'nn'
        p = subprocess.check_output('cat /proc/cpuinfo | grep name | head -n 1',
                                    shell = True)
        device_name = str(p).split(':')[1][:-3]

    print('\nRunning on device: %s' % (device_name))


    def _time():
        if args.cuda:
            torch.cuda.synchronize()

        return time.time()

    for bs in [1, 5, 8, 19]:
        for arch, sizes in arch_dict.items(): 
            if arch == 'unet3d':
                batch_size, c, d, h, w = sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]
                batch_size = bs
                print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%dx%d' %
                     (arch, kernel, batch_size, c, d, h, w))
                torch.manual_seed(0)

                data_ = torch.randn(batch_size, c, d, h, w).to_zendnn()
            else:
                batch_size, c, h, w = sizes[0], sizes[1], sizes[2], sizes[3]
                batch_size = 64 if arch == 'resnet50' and args.inference else batch_size
                batch_size = bs

                print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d' %
                     (arch, kernel, batch_size, c, h, w))

                torch.manual_seed(0)
                data_ = torch.randn(batch_size, c, h, w)

            target_ = torch.arange(1, batch_size + 1).long()

            net = models.__dict__[arch]() # no need to load pre-trained weights for dummy data

            optimizer = optim.SGD(net.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()

            if arch == 'overfeat' or arch == 'alexnet' or arch == 'vgg11':
                net.eval()

            data, target = Variable(data_), Variable(target_)

            time_fwd, time_bwd, time_upt = 0, 0, 0
            with torch.no_grad():
                steps=1
                for omp in [1, 5, 8, 24]:
                    os.environ["OMP_NUM_THREADS"] = str(omp)
                    t1 = _time()
                    output = net(data)
                    t2 = _time()
                    time_fwd = time_fwd + (t2 - t1)
                    omp = os.getenv('OMP_NUM_THREADS')
                    excepted_output = torch.load('./mkldnn_cnn_outputs/mkldnn_'+arch+'_bs_'+str(bs)+'_omp_'+str(omp)+'.pt')
                    diff = torch.max(torch.absolute(output - excepted_output))
                    if diff < 0.0001:
                        print("\n********************* output matching for ",arch," with batch size = ",bs,"for OMP_NUM_THREADS =", omp," *********************\n")
                    else:
                        print("\n********************* warning output mismatching for ",arch," with batch size = ",bs,"for OMP_NUM_THREADS =", omp," *********************\n")


                    time_fwd_avg = time_fwd / steps * 1000
                    time_bwd_avg = time_bwd / steps * 1000
                    time_upt_avg = time_upt / steps * 1000

                    # update not included!
                    time_total = time_fwd_avg + time_bwd_avg


                    print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)\n" % (kernel, ':forward:',
                          time_fwd_avg, batch_size*1000/time_fwd_avg ))


if __name__ == '__main__':
    
    # benchmark settings
    parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
    
    parser.add_argument('--arch',  action='store', default='all',
                       help='model name can be specified. all is default.' )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable CUDA')
    parser.add_argument('--inference', action='store_true', default=False,
                       help='run inference only')
    parser.add_argument('--print-iteration-time', action='store_true', default=False,
                       help='print iteration time')
                
    

    args = parser.parse_args()
    archs = OrderedDict()
    archs['alexnet'] = [1, 3, 224, 224]
    archs['vgg11'] = [1, 3, 224, 224]
    archs['googlenet'] = [1, 3, 224, 224]
    archs['overfeat'] = [1, 3, 231, 231]
    archs['inception_v3'] = [1, 3, 299, 299]
    archs['inceptionv4'] = [1, 3, 299, 299]
    archs['resnet50'] = [1, 3, 224, 224]
    archs['resnet152'] = [1, 3, 224, 224]
    archs['squeezenet1_0'] = [1, 3, 224, 224]
    archs['densenet121'] = [1, 3, 224, 224]
    archs['mobilenet_v2'] = [1, 3, 224, 224]
    archs['shufflenet'] = [1, 3, 224, 224]
    archs['unet'] = [1, 3, 128, 128]
    archs['unet3d'] = [1, 4, 64, 64, 64]

    archs_list = list(archs.keys())
    steps = 10 # nb of steps in loop to average perf
    nDryRuns = 5 # nb of warmup steps

    benchmark(args, archs_list, steps, nDryRuns)
