# resnet12 (SIM)
/home/hanlu/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 0.5 --lr 3e-4 --lr_mul 1 --lr_scheduler constant --gpu 1 --episodes_per_epoch 500 \
  --eval_interval 2

  # eval supervised pretrain(Res12 qsim cosine negative = 32, hard mining)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 0 --num_negative 32 --hard_mining
  # eval supervised pretrain(Res12 qsim cosine negative = 32, random)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1 --num_negative 32
  # eval supervised pretrain(Res12 qsim divide cosine negative = 32, random)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1 --num_negative 32 --qsim_method divide
  # eval supervised pretrain(Res12 qsim divide cosine negative = 32, hard mining)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1 --num_negative 32 --qsim_method divide \
  --hard_mining
  # eval supervised pretrain(Res12 qsim multiply cosine negative = 32, random)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 0 --num_negative 32 --qsim_method multiply
  # eval supervised pretrain(Res12 qsim multiply cosine negative = 32, hard mining)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1 --num_negative 32 --qsim_method multiply \
  --hard_mining


/home/hanlu/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet \
  --backbone_class Res12 --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 0 --episodes_per_epoch 500 --eval_interval 2 --additional ProjectionHead \
  --hidden_ratio 0.5


# eval supervised pretrain(Res12 qsim divide cosine negative = 32, random, fixres 140)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --eval_all --model_class QsimProtoNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1 --num_negative 32 --qsim_method divide \
  --test_size 140


# eval supervised pretrain(Res12 qsim divide cosine negative = 32, random, fixres 140)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --eval_all --model_class QsimMatchNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1 --num_negative 32 --qsim_method divide \
  --test_size 140


  # resnet12 (SIM lars wd 1e-4 t 0.5)
/home/hanlu/anaconda3/envs/ptg/bin/python train_fsl.py --eval_all --unsupervised --batch_size 32 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 200 --model_class ProtoNet --backbone_class Res12 \
  --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 0.5 --temperature2 0.5 --lr 0.8 --lr_mul 1 --lr_scheduler cosine --gpu 0 \
  --episodes_per_epoch 500 --eval_interval 2 --weight_decay 1e-4 --lars


  # eval supervised pretrain(fix res 140)
/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --eval_all --model_class ProtoNet --backbone_class Res12 --num_test_episodes 5000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 0 --test_size 140

/home/hanlu/anaconda3/envs/ptg/bin/python eval_fsl.py --path './checkpoints/best/Res12-pre.pth' \
  --eval_all --model_class MeanNet --backbone_class Res12 --num_test_episodes 10000 \
  --num_classes 16 --eval_way 5 --eval_shot 1 --eval_query 15 --gpu 1

# qsim pairs res12
/home/hanlu/anaconda3/envs/ptg/bin/python qsim_pairs.py -w 5 -s 1 -q 1 --backbone_class Res12 \
 --init_weights ./checkpoints/best/Res12-pre.pth

 # linear svm eval, fixres
/home/hanlu/anaconda3/envs/ptg/bin/python eval_with_svm.py --query 15 --backbone_class Res12 \
  --gpu 1 --init_weights './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --test_size 140
 # linear svm eval, fixres
/home/hanlu/anaconda3/envs/ptg/bin/python eval_simpleshot.py --backbone_class Res12 \
  --gpu 0 --model_path './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --test_size 140

   # linear svm eval, fixres , normalize
/home/hanlu/anaconda3/envs/ptg/bin/python eval_with_svm.py --query 15 --backbone_class Res12 \
  --gpu 0 --init_weights './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --test_size 140 -n
   # linear svm eval, fixres , centralize normalize
/home/hanlu/anaconda3/envs/ptg/bin/python eval_with_svm.py --query 15 --backbone_class Res12 \
  --gpu 1 --init_weights './FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth' \
  --test_size 140 -c -n