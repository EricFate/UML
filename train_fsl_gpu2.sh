# supervised from scratch (1 shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2
# supervised from scratch (1 shot SIM)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2
# supervised from scratch (1 shot SIM CUB)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --backbone_class ConvNet --dataset CUB --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2
# supervised from scratch (1 shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --num_tasks 64 --max_epoch 100 \
  --model_class DummyProto --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --dummy_samples 128 --dummy_nodes 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 1 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 \
  --eval_interval 2
# supervised from scratch (1 shot miniimagenet -> cub)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet \
  --eval_dataset CUB --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2
# supervised from scratch (1 shot, task contrastive)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 \
  --eval_interval 2 --additional TaskContrastive
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --path './checkpoints/MiniImageNet-MiniImageNet-ProtoNet-ConvNet-05w01s05q-DIS-task_contrastive/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.1_bsz096_batch032_ntask256_nclass016_ep100_evalFalse-Aug_none/max_acc.pth' \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --eval_dataset CUB \
  --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 \
  --eval_interval 2 --additional TaskContrastive

# supervised from scratch (1 shot, task contrastive)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 \
  --eval_interval 2 --additional TaskContrastive
# supervised from scratch (5 shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 5 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2
# taco
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 1)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 7 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 2)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 2 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 4)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 4 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 8)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 8 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 16)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 0 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 32)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 32 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 64)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 2 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 128)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 128 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 9 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (task 512)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 512 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (200 epoch)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 512 --max_epoch 200 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2
# taco + dummyproto
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class DummyProto \
  --dummy_samples 128 --dummy_nodes 5 \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 \
  --eval_interval 2
# taco + dummyproto dummy nodes 64
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class DummyProto \
  --dummy_samples 128 --dummy_nodes 64 \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 \
  --eval_interval 2
# taco + dummyproto dummy nodes 128
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class DummyProto \
  --dummy_samples 128 --dummy_nodes 128 \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 \
  --eval_interval 2
# taco
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2

# taco + task mixup (layer 1)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 8 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 1
# taco + task mixup (layer 2)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 11 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 2
# taco + task mixup (layer 3)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 3
# taco + task mixup (layer 4)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 7 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 4
# taco + task mixup (layer random)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 4 --rand

# taco + task mixup (layer 2 alpha 0.5)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.5 --layer 2

# taco + task mixup (layer 2 alpha 2)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 3 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 2 --layer 2

# taco + task mixup (layer 2 alpha 5)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 5 --layer 2

# taco + task mixup
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2
# taco + task mixup alpha 2
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 2
# taco + task mixup alpha 0.5
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 11 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.5

# taco + task mixup alpha 5
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 5
# taco + task gate
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional TaskGate \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 9 --episodes_per_epoch 500 \
  --eval_interval 2

# taco + task contrastive
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional TaskContrastive \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 9 --episodes_per_epoch 500 \
  --eval_interval 2
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --unsupervised --batch_size 64 \
  --path './checkpoints/tacoMiniImageNet-MiniImageNet-ProtoNet-ConvNet-05w01s05q-DIS-task_contrastive/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.1_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --additional TaskContrastive \
  --backbone_class ConvNet --dataset MiniImageNet --eval_dataset CUB \
  --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (miniimage -> CUB)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 9 --episodes_per_epoch 500 \
  --eval_interval 2
# taco (CUB miniimage)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset CUB --eval_dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 9 --episodes_per_epoch 500 \
  --eval_interval 2
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --unsupervised --batch_size 64 \
  --path './checkpoints/tacoCUB-MiniImageNet-ProtoNet-ConvNet-05w01s05q-DIS/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset CUB --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 9 --episodes_per_epoch 500 \
  --eval_interval 2

# taco(SIM, parallel bs 256, way 64)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 256 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 64 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu '8,9,10,15' --multi_gpu --episodes_per_epoch 500 --eval_interval 2
# taco(SIM, bs 64, way 32)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 32 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu '13' --episodes_per_epoch 500 --eval_interval 2
# taco(SIM, bs 64, way 64)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 64 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu '8' --episodes_per_epoch 500 --eval_interval 2
# taco(SIM, parallel bs 256, way 5)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 256 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu '10,11,14,15' --multi_gpu --episodes_per_epoch 500 --eval_interval 2
# taco(SIM, parallel bs 256, way 128)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 256 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 128 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu '10,11' --multi_gpu --episodes_per_epoch 500 --eval_interval 2
# taco(SIM, parallel bs 256, way 128, task 512)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 256 \
  --augment 'AMDIM' --num_tasks 512 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 128 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu '11,12' --multi_gpu --episodes_per_epoch 500 --eval_interval 2
# taco(SIM) extreme proto
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ExtremeProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 9 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM) memory bank proto
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM) memory bank proto (euclidean)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2 --K 10
# taco(SIM) memory bank proto (start = 50)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 6 --episodes_per_epoch 500 --eval_interval 2 --start 50
# taco(SIM) memory bank proto (5 way 1 task  taco init)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 5 \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 200 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 11 --episodes_per_epoch 500 --eval_interval 2 --K 128 --start 100
# taco(SIM) memory bank proto (max_pool)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 0 --episodes_per_epoch 500 --eval_interval 2 --max_pool
# taco(SIM) memory bank proto 闂傚倸鍊搁崐鐑芥倿閿旀拝鎷风粭娑樻噽閻瑩鏌熺�涙绠伴柤鏉挎健閺屻劑鎮ら崒娑橆伓=256闂傚倸鍊搁崐鐑芥倿閿旀拝鎷风粭娑樻搐缁�鍐ㄢ攽閻樺疇澹樼紒鐙�鍨堕弻銊╂偆閸屾稑顏�
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2 --K 256

# taco(SIM) memory bank proto (Q = 0 K = 128 200ep)
python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2 --bank_ratio 0 --K 128
# taco(SIM) memory bank proto (Q = 0 K = 128 800ep)
python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 800 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2 --bank_ratio 0 --K 128
# taco(SIM) memory bank proto (Q = 0 m = 0.9)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2 --bank_ratio 0 --K 128 --m 0.9
# taco(SIM)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 13 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM SimCLR)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'SimCLR' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 9 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM AutoAug)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AutoAug' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM 5 shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 5 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM 10 shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 10 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 9 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM projection head)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2 --additional ProjectionHead
# taco(SIM projection head)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class Res12 \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 0 --episodes_per_epoch 500 --eval_interval 2 --additional ProjectionHead \
  --hidden_ratio 0.5

/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --unsupervised --batch_size 64 \
  --path './checkpoints/tacoMiniImageNet-MiniImageNet-ProtoNet-ConvNet-05w01s05q-SIM/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 1 --model_class ProtoNet \
  --backbone_class ConvNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2
# -------------- taco same batch experiments --------------- #
# -------------------------- query shot -----------------------------

# taco(SIM 64 1 3)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 3 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 64 3 1)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 3 --query 1 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 11 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 128 1 1)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 128 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 1 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 32 1 7)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 7 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 9 --episodes_per_epoch 500 --eval_interval 2
# taco(SIM 32 7 1)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 7 --query 1 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 9 --episodes_per_epoch 500 --eval_interval 2
# -------------------------- num-class way query  -----------------------------
# taco(SIM 32 8 1 3)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 8 --shot 1 --query 3 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 8 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 32 16 1 3)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 16 --shot 1 --query 3 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 32 32 1 3)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 32 --shot 1 --query 3 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 12 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 16 8 1 7)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 16 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 8 --shot 1 --query 7 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# taco(SIM 16 16 1 7)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 16 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 16 --shot 1 --query 7 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 2 --episodes_per_epoch 500 --eval_interval 2

# -------------- taco same batch experiments --------------- #

# -------------- taco small batch (128) experiments --------------- #
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 3 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2

/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 3 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2

# -------------- taco small batch (128) experiments --------------- #

# -------------- taco number of query experiments --------------- #
# query 1
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 1 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 2 --episodes_per_epoch 500 --eval_interval 2
# query 2
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 2 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 4 --episodes_per_epoch 500 --eval_interval 2
# query 4
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 4 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 7 --episodes_per_epoch 500 --eval_interval 2
# query 8
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 8 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 8 --episodes_per_epoch 500 --eval_interval 2
# query 16
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 16 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2

# -------------- taco small batch (128) experiments --------------- #

# -------------- single task experiments --------------- #
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 5 \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2
# task 16
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 5 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 12 --episodes_per_epoch 500 --eval_interval 2
# 200 epoch
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 5 \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 200 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 13 --episodes_per_epoch 500 --eval_interval 2
# memory bank start 50
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 5 \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 11 --episodes_per_epoch 500 --eval_interval 2 --K 1024 --start 50
# memory bank start 0
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 5 \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2 --K 1024 --start 0
# memory bank start 0 (unsupervised init)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.00001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2 --K 128 --start 0

# -------------- single task experiments --------------- #

# taco(SIM , MatchNet)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 7 --episodes_per_epoch 500 --eval_interval 2
# taco (5-shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 5 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2
# taco (10-shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 10 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2
# resnet12 (SIM lars wd 1e-4 t 0.5)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 0.5 --temperature2 0.5 --lr 0.8 --lr_mul 1 --lr_scheduler cosine --gpu 2 \
  --episodes_per_epoch 500 --eval_interval 2 --weight_decay 1e-4 --lars
# resnet12 (SIM lars wd 1e-4 t 0.2)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 0.2 --temperature2 0.2 --lr 0.8 --lr_mul 1 --lr_scheduler cosine --gpu 3 \
  --episodes_per_epoch 500 --eval_interval 2 --weight_decay 1e-4 --lars

# resnet12 (10 shot)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 10 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 10 --episodes_per_epoch 500 --eval_interval 2
# cosine similarity
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised \
  --batch_size 64 --augment 'AMDIM' --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# SimCLR
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'SimCLR' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# smaller batch_size 64 -> 32
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 1 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 1 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 64 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 2 --episodes_per_epoch 500 --eval_interval 2
# supervised init 128 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 128 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 3 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' \
  --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 \
  --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine \
  --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task (50 pretrain finetune)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --init_weights './PreTrain-MiniImageNet-ConvNetLS0.0MX0.0/Bsz32Epoch-150-Cos-lr0.01decay0.0001Warmup5/max_acc_sim.pth' \
  --augment 'AMDIM' \
  --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 \
  --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine \
  --step_size 20 --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2 --finetune
# eval supervised init 256 task (50 pretrain finetune)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all \
  --path './checkpoints/MiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-finetune_50-Pre_max_acc_sim-SIM/20_0.5_lr0.0001mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/epoch-last.pth' \
  --augment 'AMDIM' \
  --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 \
  --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine \
  --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2 --finetune

/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --path ./checkpoints/MiniImageNet-MatchNet-ConvNet-05w01s05q-Pre_con_pre-DIS/20_0.5_lr0.0001mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016-Aug_AMDIM/max_acc.pth --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2

/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all \
  --path ./checkpoints/MiniImageNet-MatchNet-ConvNet-05w01s05q-Pre_con_pre-DIS/20_0.5_lr0.0001mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016-Aug_AMDIM/max_acc.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --eval_dataset CUB \
  --num_classes 16 --way 5 --eval_way 5 \
  --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 \
  --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task (protonet)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task 10 shot
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 10 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 --eval_interval 2
# supervised init 512 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 512 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 5 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task (SIM)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 7 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task conv (mixed loss)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --additional Mixed --batch_size 16 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 1 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --additional Mixed --batch_size 64 \
  --path ./checkpoints//MiniImageNet-MiniImageNet-ProtoNet-ConvNet-05w01s05q-Pre_con_pre-DIS/20_0.5_lr0.0001mul1_cosine_T11.0T21.0_b0.1_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0.1 --temperature 1 --temperature2 1 --lr 0.0001 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2

# supervised init 256 task conv (mixed loss balance 1)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --additional Mixed --batch_size 64 \
  --init_weights ./checkpoints/best/con_pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 1 --temperature 1 --temperature2 1 --lr 0.0001 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 8 --episodes_per_epoch 500 \
  --eval_interval 2
# unsupervised init 1 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth \
  --augment 'AMDIM' --num_tasks 1 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 \
  --gpu 3 --episodes_per_epoch 500 --eval_interval 2
# unsupervised init 64 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth \
  --augment 'AMDIM' --num_tasks 64 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2
# unsupervised init 128 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth \
  --augment 'AMDIM' --num_tasks 128 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 12 --episodes_per_epoch 500 --eval_interval 2
# unsupervised init 256 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 \
  --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 \
  --episodes_per_epoch 500 --eval_interval 2
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all \
  --path ./checkpoints/MiniImageNet-MatchNet-ConvNet-05w01s05q-Pre_ucon_pre-DIS/20_0.5_lr0.0001mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask128_nclass016_evalFalse-Aug_AMDIM/max_acc.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 \
  --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 \
  --episodes_per_epoch 500 --eval_interval 2
# unsupervised init 256 task 0.001 lr
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 \
  --gpu 13 --episodes_per_epoch 500 --eval_interval 2
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all \
  --path ./checkpoints/MiniImageNet-MatchNet-ConvNet-05w01s05q-Pre_ucon_pre-DIS/20_0.5_lr0.001mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_evalFalse-Aug_AMDIM/max_acc.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet \
  --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 \
  --gpu 14 --episodes_per_epoch 500 --eval_interval 2
# supervised from scratch 256 task finetune (sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 9 --episodes_per_epoch 500 --eval_interval 2 --finetune

# unsupervised init 512 task
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth --augment 'AMDIM' --num_tasks 512 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2

# -----------------------   finetuning exp ----------------------------------------------
# unsupervised init 256 task finetune 0 10 (sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2 --finetune --samples_per_class 10

# unsupervised init 256 task finetune 0 20 (sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 3 --episodes_per_epoch 500 --eval_interval 2 --finetune --samples_per_class 20

# unsupervised init 256 task finetune 0 50 (sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2 --finetune
# unsupervised init 256 task finetune 8 0 (sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 7 --episodes_per_epoch 500 --eval_interval 2 --finetune --finetune_ways 8 --samples_per_class 0
# supervised from scratch 256 task finetune 8 0 (sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 8 --episodes_per_epoch 500 --eval_interval 2 --finetune --finetune_ways 8 --samples_per_class 0
# -----------------------   finetuning exp ----------------------------------------------

# unsupervised init 256 task(sim)
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --init_weights './checkpoints/tacoMiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-SIM/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 6 --episodes_per_epoch 500 --eval_interval 2
# unsupervised init 256 task(sim)  0.001 lr
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --init_weights './checkpoints/tacoMiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-SIM/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 6 --episodes_per_epoch 500 --eval_interval 2
# unsupervised init 256 task(sim) CUB 0.002 lr
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --init_weights './checkpoints/tacoMiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-SIM/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --backbone_class ConvNet \
  --dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 10 --episodes_per_epoch 500 --eval_interval 2

# unsupervised init taco(sim) CUB
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --init_weights './checkpoints/tacoMiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-SIM/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch064_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --backbone_class ConvNet --dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2

# unsupervised init 256 task resnet12
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/uRes12-pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.01 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 14 --episodes_per_epoch 500 --eval_interval 2

# supervised init taco
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 2
/home/yangy/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --batch_size 64 --augment 'AMDIM' --num_tasks 256 --max_epoch 1 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 --eval_interval 1

# linear svm eval, unsuperivsed init
/home/yangy/anaconda3/envs/ptg/bin/python eval_with_svm.py --query 15 --backbone_class ConvNet \
  --gpu 12 --init_weights './checkpoints/best/ucon_pre_sim_proto.pth'
# linear svm eval, unsuperivsed init (Res12)
/home/yangy/anaconda3/envs/ptg/bin/python eval_with_svm.py --query 15 --backbone_class Res12 \
  --gpu 13 --init_weights './checkpoints/best/uRes_pre_sim.pth'

# visualize supervised pretrain
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --unsupervised --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet \
  --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 \
  --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 \
  --gpu 13 --episodes_per_epoch 500 --eval_interval 2

# visualize conv pretrain (qsim cosine)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 13 --mahalanobis \
  --mat_path ./checkpoints/lmnn/conv.npy

# visualize conv pretrain (qsim cosine)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class ConvNet --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 13

# visualize conv pretrain (qsim match cosine)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --model_class QsimMatchNet --backbone_class ConvNet --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 13

# eval supervised pretrain(Res12 qsim cosine negative = 1)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 8 --num_negative 1
# eval supervised pretrain(Res12 qsim cosine negative = 2)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 4 --num_negative 2
# eval supervised pretrain(Res12 qsim cosine negative = 4)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 12 --num_negative 4
# eval supervised pretrain(Res12 qsim cosine negative = 8)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 4 --num_negative 8

# eval supervised pretrain(Res12 qsim cosine negative = 16)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 5 --num_negative 16

# eval supervised pretrain(Res12 qsim match cosine negative = 16)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimMatchNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 5 --num_negative 16

# eval supervised pretrain(Res12 qsim cosine negative = 32)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 7 --num_negative 32
# eval supervised pretrain(Res12 qsim cosine negative = 64)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 7 --num_negative 64
# eval supervised pretrain(Res12 qsim cosine negative = 128)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 8 --num_negative 128
# eval supervised pretrain(Res12 qsim cosine negative = 256)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 9 --num_negative 256

# eval SimCLR pretrain
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path '/home/yangy/hanl/SimCLR/logs/1/checkpoint_100.tar' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 4

# eval SimCLR pretrain
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path '/home/yangy/hanl/SimCLR/logs/3/checkpoint_200.tar' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 4

# eval SimCLR pretrain 闂備焦瀵ч崘濠氬箯閿燂拷32 batch with head闂備焦瀵ч崘濠氬箯閿燂拷
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path '/home/yangy/hanl/SimCLR/logs/4/checkpoint_200.tar' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 3

# eval SimCLR pretrain 闂備焦瀵ч崘濠氬箯閿燂拷32 batch without head闂備焦瀵ч崘濠氬箯閿燂拷
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path '/home/yangy/hanl/SimCLR/logs/5/checkpoint_200.tar' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1

# eval SimCLR pretrain 闂備焦瀵х粙鎴濐焽瀵v 32 batch with head闂備焦瀵ч崘濠氬箯閿燂拷
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path '/home/yangy/hanl/SimCLR/logs/7/checkpoint_200.tar' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 2
# eval SimCLR pretrain 闂備焦瀵х粙鎴濐焽瀵v 32 batch without head闂備焦瀵ч崘濠氬箯閿燂拷
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path '/home/yangy/hanl/SimCLR/logs/8/checkpoint_200.tar' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 3

# eval supervised pretrain(Res12 qsim divide cosine negative = 32, hard mining)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 7 --num_negative 32 --qsim_method divide \
  --hard_mining

# visualize conv pretrain(cosine)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 10

# eval supervised pretrain(Res12 cosine)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 12
# eval supervised pretrain(fix res 140)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 12 --test_size 140

# eval supervised pretrain(Res12)
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class ProtoNet --use_euclidean --backbone_class Res12 --dataset MiniImageNet \
  --eval_dataset MiniImageNet --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 10
# visualize conv pretrain
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet \
  --eval_dataset CUB --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 13
# eval conv pretrain on CUB
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/con_pre.pth' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet \
  --eval_dataset CUB --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 4
# eval conv pretrain on CUB
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/ucon_pre.pth' \
  --eval_all --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet \
  --eval_dataset CUB --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 10
# eval conv pretrain on CUB
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class ProtoNet --use_euclidean --backbone_class Res12 --dataset MiniImageNet \
  --eval_dataset CUB --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 10
# eval conv pretrain on CUB
/home/yangy/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/uRes12-pre.pth' \
  --eval_all --model_class ProtoNet --use_euclidean --backbone_class Res12 --dataset MiniImageNet \
  --eval_dataset CUB --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 14
