# Image Classification - ImageNet

In this task, the model learns to predict the class of an image, out of 1000 classes.

## Requirements

- python >= 3.7
- python libraries:
```bash
pip install -r requirements.txt
```

## Data preparation

We use the standard ImageNet dataset, you can download it from http://image-net.org/. Validation images are put in labeled sub-folders. The file structure should look like:
```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Training from scratch
To train a `Swin Transformer` from scratch, run:
```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \ 
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

To reproduce baseline `Swin-T` results, on 1 GPU and batch size 128, run:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--batch-size 128 --data-path <imagenet-path> \
--cfg configs/swin_tiny_patch4_window7_224.yaml
```

To reproduce Hard GFiSH `Swin-T` results, on 1 GPU and batch size 128, run:
```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--batch-size 128 --data-path <imagenet-path> \
--cfg configs/swin_tiny_patch4_window7_224_hard_gfish_half.yaml
```