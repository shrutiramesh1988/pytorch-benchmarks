import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import time
import subprocess
from collections import OrderedDict
#import torch.autograd.profiler as profiler
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

    file = open(args.path, 'w')
    file.write('Running on device: %s' % (device_name))
    print('Running on device: %s' % (device_name))


    def _time():
        if args.cuda:
            torch.cuda.synchronize()

        return time.time()

    for arch, sizes in arch_dict.items(): 
        if arch == 'unet3d':
            batch_size, c, d, h, w = sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]
            batch_size = 1 if args.single_batch_size else batch_size
            file.write('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, d, h, w))
            print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, d, h, w))
            #torch.manual_seed(9)
            data_ = torch.randn(batch_size, c, d, h, w).to_zendnn()
        else:
            batch_size, c, h, w = sizes[0], sizes[1], sizes[2], sizes[3]
            batch_size = 64 if arch == 'resnet50' and args.inference else batch_size
            batch_size = 1 if args.single_batch_size else batch_size
            file.write('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, h, w))
            print('ModelType: %s, Kernels: %s Input shape: %dx%dx%dx%d' %
                 (arch, kernel, batch_size, c, h, w))
            
            torch.manual_seed(0)
            data_ = torch.randn(batch_size, c, h, w)
            #data_ = torch.Tensor(batch_size, c, h, w)
            #torch.nn.init.constant(data_, val=9.956)

        #print(data_.size())
        #print(data_)
        target_ = torch.arange(1, batch_size + 1).long()

        net = models.__dict__[arch]() # no need to load pre-trained weights for dummy data

        optimizer = optim.SGD(net.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        if arch == 'overfeat' or arch == 'alexnet' or arch == 'vgg11':
            net.eval()

        data, target = Variable(data_), Variable(target_)
        #print(data, target)
        
        for i in range(nDryRuns):
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(data)


        time_fwd, time_bwd, time_upt = 0, 0, 0
        #
        with torch.no_grad():
            for i in range(steps):
                t1 = _time()
                output = net(data)
                t2 = _time()
                time_fwd = time_fwd + (t2 - t1)
                torch.save(output, './mkldnn_cnn_outputs/mkldnn_'+arch+'_bs_'+str(bs)+'_omp_'+str(os.getenv['OMP_NUM_THREADS'])+'.pt')
        
    
        #print(output)

        time_fwd_avg = time_fwd / steps * 1000
        time_bwd_avg = time_bwd / steps * 1000
        time_upt_avg = time_upt / steps * 1000

        # update not included!
        time_total = time_fwd_avg + time_bwd_avg

        
        file.write("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':forward:',
              time_fwd_avg, batch_size*1000/time_fwd_avg ))
        file.write("%-30s %10s %10.2f (ms)" % (kernel, ':backward:', time_bwd_avg))
        file.write("%-30s %10s %10.2f (ms)" % (kernel, ':update:', time_upt_avg))
        file.write("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':total:',
              time_total, batch_size*1000/time_total ))
        
        print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':forward:',
              time_fwd_avg, batch_size*1000/time_fwd_avg ))
        print("%-30s %10s %10.2f (ms)" % (kernel, ':backward:', time_bwd_avg))
        print("%-30s %10s %10.2f (ms)" % (kernel, ':update:', time_upt_avg))
        print("%-30s %10s %10.2f (ms) %10.2f (imgs/s)" % (kernel, ':total:',
              time_total, batch_size*1000/time_total ))
        file.close()


if __name__ == '__main__':
    
    # benchmark settings
    parser = argparse.ArgumentParser(description='PyTorch Convnet Benchmark')
    
    parser.add_argument('--arch',  action='store', default='all',
                       help='model name can be specified. all is default.' )
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='disable CUDA')
    parser.add_argument('--inference', action='store_true', default=False,
                       help='run inference only')
    parser.add_argument('--single-batch-size', action='store_true', default=False,
                       help='single batch size')
    parser.add_argument('--print-iteration-time', action='store_true', default=False,
                       help='print iteration time')
    parser.add_argument('--batch_size', action='store', default=1, type=int,
                        help="batch size")
    parser.add_argument('--path', action='store', default='./cnn_log.txt', type=str,
                        help="path to store output")
                
    

    args = parser.parse_args()
    archs = OrderedDict()
    archs['alexnet'] = [args.batch_size, 3, 224, 224]
    archs['vgg11'] = [args.batch_size, 3, 224, 224]
    archs['googlenet'] = [args.batch_size, 3, 224, 224]
    archs['overfeat'] = [args.batch_size, 3, 231, 231]
    archs['inception_v3'] = [args.batch_size, 3, 299, 299]
    archs['inceptionv4'] = [args.batch_size, 3, 299, 299]
    archs['resnet50'] = [args.batch_size, 3, 224, 224]
    archs['resnet152'] = [args.batch_size, 3, 224, 224]
    archs['squeezenet1_0'] = [args.batch_size, 3, 224, 224]
    archs['densenet121'] = [args.batch_size, 3, 224, 224]
    archs['mobilenet_v2'] = [args.batch_size, 3, 224, 224]
    archs['shufflenet'] = [args.batch_size, 3, 224, 224]
    archs['unet'] = [args.batch_size, 3, 128, 128]
    archs['unet3d'] = [args.batch_size, 4, 64, 64, 64]

    archs_list = list(archs.keys())
    steps = 1 # nb of steps in loop to average perf
    nDryRuns = 1 # nb of warmup steps

    benchmark(args, archs_list, steps, nDryRuns)
