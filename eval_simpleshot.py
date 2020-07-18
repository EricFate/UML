import argparse
import os.path as osp
import pickle
import numpy as np
import torch
import shutil
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler,RandomSampler, ClassSampler
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, compute_confidence_interval
from tensorboardX import SummaryWriter
from tqdm import tqdm
from model.utils import one_hot
from model.dataloader.mini_imagenet import MiniImageNet as Dataset
from model.models.SimpleShot import SimpleShot as Model           
import sklearn.cluster
import sklearn.metrics
from sklearn.metrics.pairwise import euclidean_distances

# for Simple-Shot, we subtract the mean of SEEN classes

#  --- functional helper ---
def category_mean(data, label, label_max):
    '''compute mean for each category'''
    one_hot_label = one_hot(label, label_max)
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    return torch.mm(data.view(1, -1), one_hot_label).squeeze(0)

def category_mean2(data, label, label_max):
    '''compute mean for each category, based on a matrix'''
    one_hot_label = one_hot(label, label_max)
    data = torch.gather(data, 1, label.unsqueeze(1))
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    return torch.mm(data.view(1, -1), one_hot_label).squeeze(0)

def category_mean3(data, label, label_max):
    '''compute mean for each category, each row corresponds to an elements'''
    one_hot_label = one_hot(label, label_max)
    class_num = torch.sum(one_hot_label, 0, keepdim=True) + 1e-15
    one_hot_label = one_hot_label / class_num
    return torch.mm(one_hot_label.t(), data)

# Evaluate a provided model
def vec2str(data):
    # transform a numpy array to string
    return '['+','.join(['{:.4f}'.format(e) for e in data]) + ']'

def matrix2str(data):
    # transform a numpy matrix to string
    num_row = data.shape[0]
    output_str = []
    for i in range(num_row):
        output_str.append(vec2str(data[i,:]))
    output_str = '\n'.join(output_str)
    output_str = '[' + output_str + ']'
    return output_str

def matrix2str2(data):
    # transform a numpy matrix to string
    num_row = data.shape[0]
    output_str = []
    for i in range(num_row):
        output_str.append('['+','.join(['{}'.format(e) for e in data[i,:]]) + ']')
    output_str = '\n'.join(output_str)
    output_str = '[' + output_str + ']'
    return output_str

def printvec(s):
    print(','.join(['{:.3f}'.format(e) for e in s]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_class', type=str, default='ConvNet',
                        choices=['ConvNet', 'Res18', 'Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'CUB', 'TieredImageNet'])
    parser.add_argument('--model_path', type=str, default='../FEAT_git/saves/miniimagenet/con-pre.pth')    # './saves/initialization/miniimagenet/Res12-pre.pth'
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    args.shot = 1
    args.orig_imsize = -1
    pprint(vars(args))
    set_gpu(args.gpu)
    
    # Dataset and Data Loader
    trainset = Dataset('train', args, augment=False) 
    testset = Dataset('test', args, augment=False) 
    args.num_class = trainset.num_class        
    
    # Get Model
    model = Model(args)
    if torch.cuda.is_available():            
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.model_path)['params']
    # remove weights for FC
    if args.backbone_class == 'ConvNet':
        pretrained_dict = {'encoder.'+k:v for k, v in pretrained_dict.items()}
    # pretrained_dict = {k.replace('module',''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print(pretrained_dict.keys())
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    
    # get SEEN-class mean
    if osp.exists('./saves/stats/{}-{}-Mean.pth'.format(args.dataset, args.backbone_class)):
        class_mean = torch.load('./saves/stats/{}-{}-Mean.pth'.format(args.dataset, args.backbone_class))
    else:
        class_sampler = ClassSampler(trainset.label)
        class_loader = DataLoader(dataset=trainset, batch_sampler=class_sampler, num_workers=2, pin_memory=True) 
        class_mean = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(class_loader), ncols=50, desc='Get Mean of SEEN Classes'):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                emb = model.encoder(data)                
                class_mean.append(torch.mean(emb, dim=0))
        
        class_mean = torch.mean(torch.stack(class_mean), 0)
        torch.save(class_mean, './saves/stats/{}-{}-Mean.pth'.format(args.dataset, args.backbone_class))
    
    # shot = [1, 5, 10, 20, 30, 50]
    # FSL Test, 1-Shot, 5-Way
    num_shots = [1, 5, 10, 20, 30, 50]
    test_acc_record_sim = np.zeros((600, len(num_shots)))
    label = torch.arange(5).repeat(15)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)        
    for shot_ind, shot in enumerate(num_shots):
        few_shot_sampler = CategoriesSampler(testset.label, 600, 5, shot + 15)
        few_shot_loader = DataLoader(dataset=testset, batch_sampler=few_shot_sampler, num_workers=4, pin_memory=True)              
    
        # Get Model
        model = Model(args)
        if torch.cuda.is_available():            
            torch.backends.cudnn.benchmark = True
            model = model.cuda()
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.model_path)['params']
        if args.backbone_class == 'ConvNet':
            pretrained_dict = {'encoder.'+k:v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        model.args.shot = shot
        with torch.no_grad():
            for i, batch in tqdm(enumerate(few_shot_loader), ncols=50, desc='1-Shot 5-Way Test'):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                logits_sim = model(data, 5*shot, class_mean)                
                acc_sim = count_acc(logits_sim, label)
                test_acc_record_sim[i, shot_ind] = acc_sim            
    
    for shot_ind, shot in enumerate(num_shots):    
        m2, pm2 = compute_confidence_interval(test_acc_record_sim[:,shot_ind])
        print('Shot-{}: Test Acc Sim - {:.5f} + {:.5f}'.format(shot, m2, pm2))