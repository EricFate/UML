/home/yangy/anaconda3/envs/ptg/bin/python pretrain.py --augment 'AMDIM' \
  --max_epoch 150 --backbone_class ConvNet --dataset MiniImageNet \
   --lr 0.01 --lr_scheduler cosine --step 20 --gamma 0.1 --gpu 13 --finetune