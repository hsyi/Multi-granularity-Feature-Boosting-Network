# Multi-granularity-Feature-Boosting-Network
This repository contains experiment code for our Papaer 'A Multi-granularity Feature Boosting Network ForPerson Re-IDentification'

![MFBN](source/MFBN2-1.jpg)

## pretrained model

1. Market-1501: [google drive](https://drive.google.com/open?id=1Qu607P0ZS9ZhyMiKEhRMk6TFVOIYcvN3)
2. DukeMTMC-reID: [google drive](https://drive.google.com/open?id=1po5JNtKZ8682JF0xRFP6JUYm4Ylmbjch)
3. cuhk-np(labeled): [google drive](https://drive.google.com/open?id=1kx4zXOWDGDxv3K2TnBCEnPiICvQyng4j)
4. cuhk-np(detected): [google drive](https://drive.google.com/open?id=1_yVXV6X9fKefSpC7MFkh2I6Sh2hjQqPE)


## train：

```
	python3 main.py  --nGPU 2 --datadir /mnt/datasets/Market-1501-v15.09.15/ --batchid 16 --batchtest 32 --test_every 100 --epochs 600 --lr_scheduler warmup_10_0 --decay_type sgdr_10_2 --lr 2e-3  --loss 1*CrossEntropy+2*Triplet --margin 1.2 --save market-1501  --optimizer ADAM --amsgrad --model mfbn --random_erasing --save_models 
```

## test and evaluation:

we recommend the evaluation code in project [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch) for test and evaluation

##Acknowledgement

[1] [Person_reID_baseline_pytorch](https://github.com/layumi/Person_reID_baseline_pytorch)
[2] [MGN-pytorch](https://github.com/seathiefwang/MGN-pytorch)
