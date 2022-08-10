# ResObj in Detectron2

## Install 

```
# e.g., pytorch + cuda 11.6, detectron2
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge

pip install cython
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Running

```
# Training
CUDA_VISIBLE_DEVICES=0,1 python lazyconfig_train_net.py --config-file configs/resobj_retinanet_R_50_FPN_1x.py --num-gpus 2 

# Eval
CUDA_VISIBLE_DEVICES=0,1 python lazyconfig_train_net.py --config-file configs/resobj_retinanet_R_50_FPN_1x.py --num-gpus 2 --eval-only train.init_checkpoint=/path/to/your/checkpoint.pth
```

## Citation

```
@article{resobj,
  author    = {Joya Chen and
               Dong Liu and
               Bin Luo and
               Xuezheng Peng and
               Tong Xu and
               Enhong Chen},
  title     = {Residual objectness for imbalance reduction},
  journal   = {Pattern Recognit.},
  volume    = {130},
  pages     = {108781},
  year      = {2022},
}
```

A more detailed README.md coming soon.