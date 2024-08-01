# **CAVAE**

## Introduction

In this study, we propose a Multimodal Co-Attention-based VAE (CAVAE) deep learning framework to integrate cancer multi-omics data for clinical risk prediction.  We evaluated our approach on eight TCGA datasets. We find that (1) MAVAE outperforms traditional machine learning and recent deep learning methods; (2) Multi-modal data yields better classification performance than single-modal data; (3) The multi-head attention mechanism improves the decision-making process; (4) Clinical and genetic data are the most important modal data. 

![](https://github.com/wenwenmin/CAVAE/blob/main/CAVAE.png)

## Dataset
All the dataset, you can check ./data/Readme.md.
For ./data file, when you download from Zenodo(\url{https://zenodo.org/records/13150316}), you can see eight file of cancer dataset, all the dataset had been split into three document: train/valid/test.

## Installatioin

- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 3090 Ti)

```python
pip install -r requirements.txt
```

## Training

```python
python main.py dataset=data/brca optimizer.lr=5e-6 model.output_logits=1 model=multi_modal_pretrained_vit_lab meta.prefix_name=CXR scheduler=cosine_annealing epochs=200 meta.batch_size=50 meta.cross_validation=False meta.num_workers=20 model.transforms.img_size=384 meta.gpus=[2] meta.imbalance_handler=None optimizer.name=AdamW optimizer.lr_scheduler=None model.meta.p_visual_dropout=.0 model.meta.p_feature_dropout=1.0
```

