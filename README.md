<div align="center">

<a href="">
  <img src="https://i.ibb.co/bW3My3c/2022-02-09-02-21-08.png" style="width: 75%; height: auto;">
</a>  

**Some PyTorch Trainloop Example**
  
*My vision of training loop in PyTorch. With DDP, configs, handsome logging and more*

[![CodeFactor](https://www.codefactor.io/repository/github/sleep3r/pytorch-trainloop/badge)](https://www.codefactor.io/repository/github/sleep3r/pytorch-trainloop)
[![python](https://img.shields.io/badge/python_3.8-passing-success)](https://github.com/sleep3r/garrus/badge.svg?branch=main&event=push)

</div>

----

## Training:

#### Distributed:

```bash
CUDA_VISIBLE_DEVICES=0,1 ./dist_train.sh --config=./configs/baseline.yml
```

#### Single GPU:

```bash
python train.py --config=./configs/baseline.yml
```
