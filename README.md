
# DRR-Weather

*Zixuan Li , Fang Long , Wenkang Su , Yuan-Gen Wang , Qingxiao Guan , Lei Cai*

Official Repository for "DRR: A New Method for Multiple Adverse Weather Removal" (Accepted by Expert Systems with Applications)



# Usage

## Dataset Preparation

We train and evaluate the proposed method on the [Cityscapes-RF](https://www.cityscapes-dataset.com/downloads/) and [Cityscapes-RFS&RFD](https://github.com/JunlinHan/BID) datasets.



## Environment setup

Preparing the environment:

```
conda create -n DRRWeather python=3.8
conda activate DRRWeather

conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -r requirements.txt
```



## Training

Run the training script:

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS train.py --opt path/to/option.json --dist True
```



## Evaluation

Run the evaluation script:

```
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS test.py --opt path/to/option.json --dist True
```



## Acknowledgments

The code is developed based on [SwinIR]([https://github.com/JingyunLiang/SwinIR) and [GT-RAIN](https://github.com/UCLA-VMG/GT-RAIN). Sincere thanks to their wonderful works.



## License

This source code is made available for research purpose only.