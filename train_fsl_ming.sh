# supervised from scratch (1 shot)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2
# supervised from scratch (5 shot)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --num_tasks 256 --max_epoch 100 \
  --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 \
  --eval_way 5 --shot 5 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 \
  --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'AMDIM' --num_tasks 64 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# resnet12
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# resnet12 (1 shot miniimagenet -> cub)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 \
  --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
  # resnet12 (SIM, cosine anealing)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler cosine \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# resnet12 (500 epoch)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 500 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 350,400,450,475 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# resnet12 (5 shot)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 5 --eval_shot 1 --query 1 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 350,400,450,475 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --init_weights ./checkpoints/best/con_pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MatchNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 7 --episodes_per_epoch 500 --eval_interval 2

# supervised init 256 task resnet12 (mixed loss balance 0.1)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --additional Mixed \
  --init_weights ./checkpoints/best/Res12-pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MatchNet --use_euclidean \
  --backbone_class Res12 --batch_size 16 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0.1 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
/home/amax/anaconda3/envs/ming/bin/python eval_fsl.py --eval_all --additional Mixed \
  --path 'checkpoints/MiniImageNet-MiniImageNet-MatchNet-Res12-05w01s05q-Pre_Res12-pre-DIS/150_175_190_0.1_lr0.0001mul1_multistep_T11.0T21.0_b0.1_bsz096_batch016_ntask256_nclass016_ep200_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MatchNet --use_euclidean \
  --backbone_class Res12 --batch_size 16 --dataset MiniImageNet --eval_dataset CUB \
  --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0.1 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# supervised init 256 task resnet12 (mixed loss balance 1)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --additional Mixed \
  --init_weights ./checkpoints/best/Res12-pre.pth \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class MatchNet --use_euclidean \
  --backbone_class Res12 --batch_size 16 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 1 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
# resnet12 (1 shot miniimagenet -> cub)
/home/amax/anaconda3/envs/ming/bin/python eval_fsl.py --eval_all --batch_size 32 \
  --path './checkpoints/MiniImageNet-CUB-ProtoNet-Res12-05w01s05q-DIS/150_175_190_0.1_lr0.1mul1_multistep_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_ep200_evalFalse-Aug_AMDIM/max_acc.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --use_euclidean --backbone_class Res12 \
  --dataset MiniImageNet --eval_dataset CUB --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 \
  --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.1 --lr_mul 1 --lr_scheduler multistep --step_size 150,175,190 \
  --gamma 0.1 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 --augment 'SimCLR' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 128 --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2

# taco + task mixup (layer 1)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 0 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 1
# taco + task mixup (layer 2)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 12 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 2
# taco + task mixup (layer 3)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 13 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 3
# taco + task mixup (layer 4)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 7 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 4
# taco + task mixup (layer random)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --additional MixUp \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 15 --episodes_per_epoch 500 \
  --eval_interval 2 --alpha 0.2 --layer 4 --rand


# unsupervised init 256 task finetune (sim)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.0001 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 1 --episodes_per_epoch 500 --eval_interval 2 --finetune
# unsupervised pretrain 256 task finetune (sim)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all --init_weights './checkpoints/best/ucon_pre_sim_proto.pth' \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 0 --episodes_per_epoch 500 --eval_interval 2 --finetune
# supervised from scratch 256 task finetune (sim)
/home/amax/anaconda3/envs/ming/bin/python train_fsl.py --eval_all \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --backbone_class ConvNet \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 0 --episodes_per_epoch 500 --eval_interval 2 --finetune

# eval last unsupervised pretrain 256 task finetune (sim)
/home/amax/anaconda3/envs/ming/bin/python eval_fsl.py \
  --path './checkpoints/MiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-Pre_ucon_pre_sim_proto-SIM/20_0.5_lr0.0001mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/epoch-last.pth' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 0


# eval last unsupervised pretrain 256 task finetune (sim)
/home/amax/anaconda3/envs/ming/bin/python eval_fsl.py \
  --path './checkpoints/MiniImageNet-MiniImageNet,CUB-ProtoNet-ConvNet-05w01s05q-none-Pre_ucon_pre_sim_proto-SIM/20_0.5_lr0.002mul1_cosine_T11.0T21.0_b0.0_bsz096_batch032_ntask256_nclass016_ep100_evalFalse-Aug_AMDIM/epoch-last.pth' \
  --eval_all --model_class ProtoNet --backbone_class ConvNet --dataset MiniImageNet \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1


/home/amax/anaconda3/envs/ming/bin/python eval_fsl.py --eval_all --path checkpoint_0199.pth.tar \
--augment moco --model_class ProtoNet --backbone_class ConvNet --num_classes 16 --way 5 \
--eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 --gpu 1

# moco
python main_moco.py \
  -a ConvNet \
  --lr 0.03 \
  --epochs 800 \
  --batch-size 256 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos