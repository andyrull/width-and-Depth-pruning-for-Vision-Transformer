# Width and Depth Pruning for Vision Transformers
This is the official implementation of the AAAI 2022 paper [Width and Depth Pruning for Vision Transformers] (https://www.aaai.org/AAAI22Papers/AAAI-2102.FangYu.pdf)



## Installation

### Requirements

- torch>=1.8.0
- torchvision>=0.9.0
- timm==0.4.9
- h5py
- scipy
- scikit-learn

Data preparation: download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

Model preparation: download pre-trained DeiT models for pruning:
```
sh download_pretrain.sh
```

## Demo

### Training on ImageNet

To train DeiT models on ImageNet, run:

DeiT-Base
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port 29500 --use_env main_wdpruning.py --arch deit_base --data-set IMNET --batch-size 128 --data-path ../data/ILSVRC2012/ --output_dir logs --classifier 10 --R_threshold 0.8
```

### Training on CIFAR-10

To train DeiT models on CIFAR-10, run:

Pruning width and depth for DeiT-Base
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29566 --use_env main_wdpruning.py --arch deit_base --data-set CIFAR10 --batch-size 128 --data-path ../data/ --output_dir logs/cifar --classifiers 10 
```

Only pruning width for DeiT-Base 
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port 29566 --use_env main_wdpruning.py --arch deit_base --data-set CIFAR10 --batch-size 128 --data-path ../data/ --output_dir logs/cifar
```



### Pruning and Evaluation
Test the amout of parameters, GPU throughput of pruned transformer. 
```
python masked_parameter_count.py --arch deit_base --pretrained_dir logs/checkpoint.pth --eval_batch_size 1024 --classifiers 10 --classifier_choose 10
```
Note that '--classifier_choose' means choose which classifier to prune. '--classifier_choose 12' means choose the last classifier. 

\
\
Test the amout of parameters, CPU latency of pruned transformer.
```
python masked_parameter_count.py --arch deit_base --pretrained_dir logs/checkpoint.pth --no_cuda --eval_batch_size 1  --classifiers 10
```

## Acknowledgement
Our code is built on top of Movement Pruning.


## Citing
If you find these useful for your research or project, feel free to cite our paper.
```
@inproceedings{yu2022width,
  title={Width \& Depth Pruning for Vision Transformers},
  author={Yu, Fang and Huang, Kun and Wang, Meng and Cheng, Yuan and Chu, Wei and Cui, Li},
  booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
  volume={2022},
  year={2022}
}
```
