import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import subprocess
from collections import OrderedDict

from mobilenet import MobileNetV2
models.__dict__['mobilenet_v2'] = MobileNetV2

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

archs = OrderedDict()
archs['alexnet'] = [1, 3, 224, 224]
archs['vgg11'] = [1, 3, 224, 224]
archs['googlenet'] = [1, 3, 224, 224]
archs['overfeat'] = [1, 3, 231, 231]
archs['inception_v3'] = [1, 3, 299, 299]
archs['inceptionv4'] = [1, 3, 299, 299]
archs['resnet50'] = [1, 3, 224, 224]
archs['resnet152'] = [1, 3, 224, 224]
archs['squeezenet1_0'] = [128, 3, 224, 224]
archs['densenet121'] = [32, 3, 224, 224]
archs['mobilenet_v2'] = [128, 3, 224, 224]
archs['shufflenet'] = [128, 3, 224, 224]
archs['unet'] = [32, 3, 128, 128]
archs['unet3d'] = [6, 4, 64, 64, 64]

archs_list = list(archs.keys())
steps = 10 # nb of steps in loop to average perf
nDryRuns = 5 # nb of warmup steps


def benchmark():
    # benchmark settings
    parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
    parser.add_argument('--arch',  action='store', default='all',
                       choices=archs_list + ['all'],
                       help='model name can be specified. all is default.' )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable CUDA')
    parser.add_argument('--inference', action='store_true', default=False,
                       help='run inference only')
    parser.add_argument('--single-batch-size', action='store_true', default=False,
                       help='single batch size')
    parser.add_argument('--print-iteration-time', action='store_true', default=False,
                       help='print iteration time')

    args = parser.parse_args()
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

    print('Running on device: %s' % (device_name))

    def _time():
        if args.cuda:
            torch.cuda.synchronize()

        return time.time()

    for arch, sizes in arch_dict.items(): 
        if arch == 'unet3d':
            batch_size, c, d, h, w = sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]
            batch_size = 1 if args.single_batch_size else batch_size
            print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, d, h, w))
            #torch.manual_seed(9)
            data_ = torch.randn(batch_size, c, d, h, w)
        else:
            batch_size, c, h, w = sizes[0], sizes[1], sizes[2], sizes[3]
            batch_size = 64 if arch is 'resnet50' and args.inference else batch_size
            batch_size = 1 if args.single_batch_size else batch_size
            print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, h, w))
            #torch.manual_seed(9)
            data_ = torch.randn(batch_size, c, h, w)
            #data_ = torch.Tensor(batch_size, c, h, w)
            #torch.nn.init.constant(data_, val=9.956)

        #print(data_.size())
        #print(data_)
        target_ = torch.arange(1, batch_size + 1).long()

        net = models.__dict__[arch]() # no need to load pre-trained weights for dummy data

        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        if args.cuda:
            data_, target_ = data_.cuda(), target_.cuda()
            net.cuda()
            criterion = criterion.cuda()

        if args.inference:
            net.eval()
        else:
            net.train()
            net.aux_logits = False

        data, target = Variable(data_), Variable(target_)

        for i in range(nDryRuns):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(data)
            if not args.inference:
                loss = output.sum() / 1e6 if 'unet' in arch else criterion(output, target)
                loss.backward()
                optimizer.step()    # Does the update

        time_fwd, time_bwd, time_upt = 0, 0, 0

        for i in range(steps):
            optimizer.zero_grad()   # zero the gradient buffers
            t1 = _time()
            output = net(data)

            #if i==0:
             #Print the OutPut Tensor
             #print(output.size())
             #print(output)

             #out = open('outfile.txt', 'w')
             #print(output, file = out)
             #out.close()

            t2 = _time()
            if not args.inference:
                loss = output.sum() / 1e6 if 'unet' in arch else criterion(output, target)
                loss.backward()
                t3 = _time()
                optimizer.step()    # Does the update
                t4 = _time()
            time_fwd = time_fwd + (t2 - t1)
            if args.print_iteration_time:
                print("%-30s %d: %10.2f ms" % ('forward iteration', i, (t2-t1)*1000))
            if not args.inference:
                time_bwd = time_bwd + (t3 - t2)
                time_upt = time_upt + (t4 - t3)

        time_fwd_avg = time_fwd / steps * 1000
        time_bwd_avg = time_bwd / steps * 1000
        time_upt_avg = time_upt / steps * 1000

        # update not included!
        time_total = time_fwd_avg + time_bwd_avg

        print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':forward:',
              time_fwd_avg, batch_size*1000/time_fwd_avg ))
        print("%-30s %10s %10.2f (ms)" % (kernel, ':backward:', time_bwd_avg))
        print("%-30s %10s %10.2f (ms)" % (kernel, ':update:', time_upt_avg))
        print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':total:',
              time_total, batch_size*1000/time_total ))

if __name__ == '__main__':
    benchmark()
