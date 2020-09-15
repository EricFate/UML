# supervised from scratch (1 shot)
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
--model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
--eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
--lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
--eval_interval 2
# supervised from scratch (5 shot)
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
--model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
--eval_way 5 --shot 5 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
--lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
--eval_interval 2

# resnet12 (SIM)
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# resnet12（CUB -> miniimagenet）
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset CUB --eval_dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
/home/amax/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all --unsupervised --batch_size 32 \
  --path './checkpoints/tacoCUB-MiniImageNet-ProtoNet-Res12-05w01s05q-DIS/150_175_190_0.1_lr0.1mul1_multistep_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_ep200_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset CUB --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2


# resnet12 (5 shot)
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
--augment 'AMDIM' --num_tasks 256  --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
--dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 5 --eval_shot 1 --query 1 --eval_query 15 \
--balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 350,400,450,475 \
--gamma 0.1  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# resnet12 (AutoAug)
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
--augment 'AutoAug' --num_tasks 256  --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
--dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
--balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
--gamma 0.1  --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# supervised train from scratch
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --num_tasks 256  --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# unsupervised train from scratch
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# cosine similarity
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# SimCLR
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'SimCLR' --num_tasks 256  --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# smaller batch_size 64 -> 32
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 1 task
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 64 task
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 128 task
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# supervised init 256 task resnet12
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all \
  --init_weights ./checkpoints/best/Res12-pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MatchNet --use_euclidean \
  --backbone_class Res12 --batch_size 16 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0.1 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
/home/amax/anaconda3/envs/ptg/bin/python eval_fsl.py --eval_all \
  --path './checkpoints/MiniImageNet-MiniImageNet-MatchNet-Res12-05w01s05q-Pre_Res12-pre-DIS/150_175_190_0.1_lr0.0001mul1_multistep_T11.0T21.0_b0.1_bsz096_batch016_ntask256_nclass016_ep200_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MatchNet --use_euclidean \
  --backbone_class Res12 --batch_size 16 --dataset MiniImageNet --eval_dataset CUB\
  --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0.1 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# supervised init 256 task resnet12 (mixed loss)
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --mixed \
  --init_weights ./checkpoints/best/con_pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MatchNet --use_euclidean \
  --backbone_class Res12 --batch_size 16 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0.1 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2


# eval res12 cub -> imagenet
/home/amax/anaconda3/envs/ptg/bin/python eval_fsl.py \
--path './checkpoints/tacoCUB-MiniImageNet-ProtoNet-Res12-05w01s05q-DIS/150_175_190_0.1_lr0.1mul1_multistep_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_ep200_evalFalse-Aug_AMDIM/max_acc.pth' \
--eval_all --model_class ProtoNet --use_euclidean --backbone_class Res12 --dataset MiniImageNet \
--eval_dataset MiniImageNet --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1

# eval res12_500 imagenet -> CUB
/home/amax/anaconda3/envs/ptg/bin/python eval_fsl.py \
--path './checkpoints/best/uRes12-pre500.pth' \
--eval_all --model_class ProtoNet --use_euclidean --backbone_class Res12 --dataset MiniImageNet \
--eval_dataset CUB --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1

# unsupervised init
/home/amax/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/ucon_pre.pth --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class MatchNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2






/home/amax/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/ucon_pre.pth' --eval_all --unsupervised --batch_size 128 --augment 'AMDIM' --num_tasks 256  --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5  --gpu 1 --episodes_per_epoch 500 --eval_interval 2