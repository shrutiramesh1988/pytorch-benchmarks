#!/bin/sh

export PATH=~/anaconda3/bin:$PATH
eval "$(conda shell.bash hook)"

PTH="./"
PTH1="./results/21_04_21"

unset OMP_NUM_THREADS
unset GOMP_CPU_AFFINITY
unset MKL_DEBUG_CPU_TYPE
unset KMP_AFFINITY
unset LD_LIBRARY_PATH
# "pytorch-1.7-zendnn-env" "pytorch-1.7-mkldnn-env" "pytorch-1.7-zendnn-env-vanilla"
for ENVI in "pytorch-1.7-mkldnn-env"
do
    if [ $ENVI ==  "pytorch-1.7-mkldnn-env" ] ;
    then
        echo "$ENVI"
        unset LD_LIBRARY_PATH
        export KMP_AFFINITY=granularity=fine,compact,1,0
        export MKL_DEBUG_CPU_TYPE=5
        #export MKL_VERBOSE=1
        #export MKLDNN_VERBOSE=1
    fi
    conda activate $ENVI
    cd $PTH
    for BS in 1 5 8 19
    do
        #"alexnet" "googlenet" "overfeat" "vgg11" "resnet50" "resnet152" "inceptionv4"
        for NET in "alexnet" "googlenet" "overfeat" "vgg11" "resnet50" "resnet152" "inceptionv4" "mobilenet_v2" "shufflenet" "unet" "unet3d"
        do
            mkdir -p $PTH1/CNN/$ENVI/$BS/$NET
            #if [ $BS ==  1 ]
            #then
                for OMP in 1 5 8 24
                do
                    export GOMP_CPU_AFFINITY=64-$(($OMP-1+64)) && export OMP_NUM_THREADS=$OMP && numactl --cpunodebind=4-7 --interleave=4-7 python $PTH/benchmark-optimized-generate-output.py --batch_size=$BS --arch=$NET 2>&1 | tee $PTH1/CNN/$ENVI/$BS/$NET/cnn_log_omp_"$OMP".txt
                done
        done
    done
    conda deactivate
done
