<div align="center">

**Some PyTorch Trainloop Example**
  
*My vision of training loop in PyTorch. With DDP, configs, handsome logging and more*

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
