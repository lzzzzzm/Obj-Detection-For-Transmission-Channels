简体中文 | [English](README_en.md)

<div align="center">
<p align="center">
  <img src="https://user-images.githubusercontent.com/48054808/160532560-34cf7a1f-d950-435e-90d2-4b0a679e5119.png" align="middle" width = "800" />
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PaddleDetection?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/PaddleDetection/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PaddleDetection?color=ccf"></a>
</p>
</div>

## Getting Start

converter to coco
```python
!python tools/x2coco.py \
    --dataset_type voc \
    --voc_anno_dir dataset/channel_transmission \
    --voc_anno_list dataset/channel_transmission/train.txt \
    --voc_label_list dataset/channel_transmission/label_list.txt \
    --voc_out_name dataset/channel_transmission/annotations/train.json
```
```python
!python tools/x2coco.py \
    --dataset_type voc \
    --voc_anno_dir dataset/channel_transmission \
    --voc_anno_list dataset/channel_transmission/train.txt \
    --voc_label_list dataset/channel_transmission/label_list.txt \
    --voc_out_name dataset/channel_transmission/annotations/train.json
```

## output model
```python
!python tools/export_model.py -c configs/dino/dino_r50_4scale_2x_coco.yml --output_dir=./inference_model -o weights=output/dino_r50_4scale_2x_coco/best_model
```

## Infer & Output json results
```python
!python deploy/python/infer.py --model_dir=inference_model/dino_r50_4scale_2x_coco --image_dir=dataset/val/ --device=GPU --output_dir infer_output --save_results
```



