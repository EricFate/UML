# taco(SIM projection head)
python  train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 15 --episodes_per_epoch 500 --eval_interval 2 --additional ProjectionHead


# taco
python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 16 --max_epoch 100 --model_class ProtoNet --use_euclidean \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --eval_way 5 --shot 1 \
  --eval_shot 1 --query 5 --eval_query 15 --balance 0 --temperature 1 --temperature2 1 --lr 0.002 \
  --lr_mul 1 --lr_scheduler cosine --step_size 20 --gamma 0.5 --gpu 14 --episodes_per_epoch 500 \
  --eval_interval 2

# taco(SIM) memory bank proto (Q = 0)
python train_fsl.py --eval_all --unsupervised --batch_size 64 \
  --augment 'AMDIM' --num_tasks 256 --max_epoch 100 --model_class MemoryBankProto \
  --backbone_class ConvNet --dataset MiniImageNet --num_classes 16 --way 5 --shot 1 --query 5 --eval_query 15 \
  --balance 1.0 --temperature 1 --temperature2 1 --lr 0.002 --lr_mul 1 --lr_scheduler cosine --step_size 20 \
  --gamma 0.5 --gpu 13 --episodes_per_epoch 500 --eval_interval 2 --bank_ratio 0