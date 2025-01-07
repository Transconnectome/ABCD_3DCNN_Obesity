## ======= load module ======= ##
import models.simple3d as simple3d #model script
import models.vgg3d as vgg3d #model script
import models.resnet3d as resnet3d #model script
import models.densenet3d as densenet3d #model script
import models.efficientnet3d as efficientnet3d
from utils.utils import set_random_seed, CLIreporter, save_exp_result, checkpoint_save, checkpoint_load, MOPED_network
from utils.lr_scheduler import *
from utils.early_stopping import * 
from dataloaders.dataloaders import check_study_sample, loading_images, loading_phenotype, combining_image_target, partition_dataset, matching_partition_dataset, undersampling_ALLset, matching_undersampling_ALLset, partition_dataset_predefined
from dataloaders.preprocessing import preprocessing_cat, preprocessing_num
from envs.experiments import train, validate, test 
import hashlib
import datetime




import os
from os import listdir
from os.path import isfile, join
from tqdm.auto import tqdm ##progress
import time
import math
import random
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from utils.optimizer import SGDW
from torchsummary import summary


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

import random
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")


def argument_setting():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",default=1234,type=int,required=False,help='')
    #parser.add_argument("--GPU_NUM",default=1,type=int,required=True,help='')
    parser.add_argument("--study_sample",default='UKB',type=str,required=False,help='')
    parser.add_argument("--model",required=True,type=str,help='',choices=[
                                                                        'resnet3D18', 'resnet3D50', 'resnet3D101','resnet3D152', 
                                                                        'densenet3D121', 'densenet3D169','densenet3D201','densenet3D264', 
                                                                        'densenet3D121_cbam', 'densenet3D169_cbam','densenet3D201_cbam','densenet3D264_cbam', 
                                                                        'flipout_densenet3D121', 'flipout_densenet3D169','flipout_densenet3D201','flipout_densenet3D264', 
                                                                        'variational_densenet3D121', 'variational_densenet3D169','variational_densenet3D201','variational_densenet3D264', 
                                                                        'efficientnet3D-b0','efficientnet3D-b1','efficientnet3D-b2','efficientnet3D-b3','efficientnet3D-b4','efficientnet3D-b5','efficientnet3D-b6','efficientnet3D-b7',
                                                                        'vit_base_patch16_3D','vit_large_patch16_3D','vit_huge_patch14_3D','vit_base_patch16_3D','vit_large_patch16_3D','vit_huge_patch14_3D'

                                                                        ])
    parser.add_argument("--train_size",default=0.8,type=float,required=False,help='')
    parser.add_argument("--val_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--test_size",default=0.1,type=float,required=False,help='')
    parser.add_argument("--resize",default=[96, 96, 96],type=int,nargs="*",required=False,help='')
    parser.add_argument("--batch_size",default=16,type=int,required=False,help='')
    parser.add_argument("--in_channels",default=1,type=int,required=False,help='')
    parser.add_argument("--optim",type=str,required=True,help='', choices=['Adam','AdamW','SGD', 'SGDW'])
    parser.add_argument("--lr", default=0.01,type=float,required=False,help='')
    parser.add_argument("--weight_decay",default=0.001,type=float,required=False,help='')
    parser.add_argument("--beta",default=1.0,type=float,required=False,help='')
    parser.add_argument("--warmup_epoch",type=int,required=False, default=5, help='')
    parser.add_argument("--warmup_interval",type=int,required=False, default=0, help='')
    parser.add_argument("--epoch",type=int,required=True,help='')
    parser.add_argument("--exp_name",type=str,required=True,help='')
    parser.add_argument("--cat_target", type=str, nargs='*', required=False, help='')
    parser.add_argument("--num_target", type=str,nargs='*', required=False, help='')
    parser.add_argument("--confusion_matrix", type=str, nargs='*',required=False, help='')
    parser.add_argument("--gpus", type=int,nargs='*', required=False, help='')
    parser.add_argument('--accumulation_steps', default=1, type=int, required=False)
    parser.add_argument("--checkpoint_dir", type=str, default=None,required=False)
    parser.add_argument('--get_predicted_score', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(get_predicted_score=False)
    parser.add_argument('--matching_baseline_2years', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(matching_baseline_2years=False)
    parser.add_argument('--matching_baseline_gps', action='store_true', help='save the result of inference in the result file')
    parser.set_defaults(matching_baseline_gps=False)
    parser.add_argument("--conv_unfreeze_iter", type=int, default=None,required=False)
    parser.add_argument('--gradient_clipping', action='store_true')
    parser.set_defaults(gradient_accumulation=False)
    parser.add_argument("--undersampling_dataset_target", type=str, default=None, required=False, help='')
    parser.add_argument('--mixup', type=float, default=0, help='')
    parser.add_argument('--cutmix', type=float, default=0, help='')
    parser.add_argument('--c_mixup', type=float, default=0, help='')
    parser.add_argument('--manifold_mixup', type=float, default=0, help='')
    parser.add_argument('--partitioned_dataset_number', default=None, type=int, required=False)
    parser.add_argument('--moped', action='store_true')
    parser.set_defaults(moped=False)
    parser.add_argument('--finetune_undersample', action='store_true')
    parser.set_defaults(finetune_undersample=False)
    parser.add_argument('--only_acc', action='store_true')
    parser.set_defaults(only_acc=False)

    args = parser.parse_args()
    print("Categorical target labels are {} and Numerical target labels are {}".format(args.cat_target, args.num_target))

    if not args.cat_target:
        args.cat_target = []
    elif not args.num_target:
        args.num_target = []
    elif not args.cat_target and args.num_target:
        raise ValueError('YOU SHOULD SELECT THE TARGET!')

    return args


## ========= Experiment =============== ##
def experiment(partition, subject_data, save_dir, args): #in_channels,out_dim
    targets = args.cat_target + args.num_target

    # ResNet
    if args.model == 'resnet3D18':
        import models.resnet3d as resnet3d
        from envs.experiments import train, validate, test 
        net = resnet3d.resnet3D18(subject_data, args)
    if args.model == 'resnet3D50':
        import models.resnet3d as resnet3d #model script
        from envs.experiments import train, validate, test 
        net = resnet3d.resnet3D50(subject_data, args)
    elif args.model == 'resnet3D101':
        import models.resnet3d as resnet3d #model script
        from envs.experiments import train, validate, test 
        net = resnet3d.resnet3D101(subject_data, args)
    elif args.model == 'resnet3D152':
        import models.resnet3d as resnet3d #model script
        from envs.experiments import train, validate, test 
        net = resnet3d.resnet3D152(subject_data, args)
    # DenseNet
    elif args.model == 'densenet3D121':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'densenet3D169':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D169(subject_data, args) 
    elif args.model == 'densenet3D201':
        import models.densenet3d as densenet3d #model script
        from envs.experiments import train, validate, test 
        net = densenet3d.densenet3D201(subject_data, args)
    # DenseNet with CBAM module
    if args.model == 'densenet3D121_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D121_cbam(subject_data, args)
    elif args.model == 'densenet3D169_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D169_cbam(subject_data, args) 
    elif args.model == 'densenet3D201_cbam':
        import models.densenet3d_cbam as densenet3d_cbam #model script
        from envs.experiments import train, validate, test 
        net = densenet3d_cbam.densenet3D201_cbam(subject_data, args)
    # Bayesian (variational) DenseNet
    elif args.model == 'variational_densenet3D121':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = variational_densenet3d.densenet3D121(subject_data, args)
            det_net = densenet3d.densenet3D121(subject_data, args)
        else: 
            net = variational_densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'variational_densenet3D169':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = variational_densenet3d.densenet3D169(subject_data, args)
            det_net = densenet3d.densenet3D169(subject_data, args)
        else: 
            net = variational_densenet3d.densenet3D169(subject_data, args)
    elif args.model == 'variational_densenet3D201':
        import models.variational_densenet3d as variational_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = variational_densenet3d.densenet3D201(subject_data, args)
            det_net = densenet3d.densenet3D201(subject_data, args)
        else: 
            net = variational_densenet3d.densenet3D201(subject_data, args)
    # Bayesian (flipout) DenseNet
    elif args.model == 'flipout_densenet3D121':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = flipout_densenet3d.densenet3D121(subject_data, args)
            det_net = densenet3d.densenet3D121(subject_data, args)
        else: 
            net = flipout_densenet3d.densenet3D121(subject_data, args)
    elif args.model == 'flipout_densenet3D169':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = flipout_densenet3d.densenet3D169(subject_data, args)
            det_net = densenet3d.densenet3D169(subject_data, args)
        else: 
            net = flipout_densenet3d.densenet3D169(subject_data, args)
    elif args.model == 'flipout_densenet3D201':
        import models.flipout_densenet3d as flipout_densenet3d
        from envs.bayesian_experiments import train, validate, test 
        if args.moped: 
            import models.densenet3d as densenet3d
            bayes_net = flipout_densenet3d.densenet3D201(subject_data, args)
            det_net = densenet3d.densenet3D201(subject_data, args)
        else: 
            net = flipout_densenet3d.densenet3D201(subject_data, args)
    # EfficientNet V1 
    elif args.model.find('efficientnet3D') != -1: 
        from envs.experiments import train, validate, test 
        net = efficientnet3d.efficientnet3D(subject_data,args)
    # ViT
    elif args.model.find('vit_') != -1:
        import models.model_ViT as ViT
        from envs.experiments import train, validate, test
        net = ViT.__dict__[args.model](img_size = args.resize, attn_drop=0.5, drop=0.5, drop_path=0.1, global_pool=True, 
                                       subject_data=subject_data, args=args)

    # load checkpoint
    if args.moped: 
        assert args.checkpoint_dir is not None 
        det_net = checkpoint_load(det_net, args.checkpoint_dir, layers='conv') 
        net = MOPED_network(bayes_net=bayes_net, det_net=det_net)   # weights of FC layer does not used for prior of Bayesian DNN if checkpoint load only convolution layers
        del det_net
        print("Prior of Bayesian DNN are set with parameters from Deterministic DNN")
    else: 
        if args.checkpoint_dir is not None: 
            net = checkpoint_load(net, args.checkpoint_dir, layers='conv')


    if args.optim == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)
    elif args.optim == 'SGDW':
        optimizer = SGDW(net.parameters(), lr=0, momentum=0.9)
    elif args.optim == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.optim == 'AdamW': 
        optimizer = optim.AdamW(net.parameters(), lr=0, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    else:
        raise ValueError('In-valid optimizer choice')

    # learning rate schedluer
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max', patience=20) #if you want to use this scheduler, you should activate the line 134 of envs/experiments.py
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=2, eta_max=args.lr, T_up=5, gamma=0.5)
    if args.warmup_interval == 0:
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.epoch, T_mult=2, eta_max=args.lr, T_up=args.warmup_epoch, gamma=0.5)
    else: 
        scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=args.warmup_interval, T_mult=2, eta_max=args.lr, T_up=args.warmup_epoch, gamma=0.5)
    # apply early stopping 
    early_stopping = EarlyStopping(patience=30)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    # pytorch 2.0 
    #net = torch.compile(net)
    
    # attach network and optimizer to cuda device
    net.cuda()


    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(f'cuda:{net.device_ids[0]}')
    """
    
    # setting for results' data frame
    train_losses = {}
    train_accs = {}
    val_losses = {}
    val_accs = {}

    for target_name in targets:
        train_losses[target_name] = []
        train_accs[target_name] = [-10000.0]
        #train_accs[target_name] = [10000.0]
        val_losses[target_name] = []
        val_accs[target_name] = [-10000.0]
        #val_accs[target_name] = [10000.0]
        
    global_steps = 0
    for epoch in tqdm(range(args.epoch)):
        ts = time.time()
        net, train_loss, train_acc, global_steps = train(net,partition,optimizer, global_steps, args)
        torch.cuda.empty_cache()
        val_loss, val_acc = validate(net,partition,scheduler,args)
        te = time.time()

         # sorting the results
        if args.cat_target: 
            for cat_target in args.cat_target: 
                train_losses[cat_target].append(train_loss[cat_target])
                train_accs[cat_target].append(train_acc[cat_target]['ACC'])
                val_losses[cat_target].append(val_loss[cat_target])
                val_accs[cat_target].append(val_acc[cat_target]['ACC'])
                early_stopping(val_acc[cat_target]['ACC'])
        if args.num_target: 
            for num_target in args.num_target: 
                train_losses[num_target].append(train_loss[num_target])
                train_accs[num_target].append(train_acc[num_target]['r_square'])
                #train_accs[num_target].append(train_acc[num_target]['abs_loss'])
                val_losses[num_target].append(val_loss[num_target])
                val_accs[num_target].append(val_acc[num_target]['r_square'])
                #val_accs[num_target].append(val_acc[num_target]['abs_loss'])
                early_stopping(val_acc[num_target]['r_square'])
                #early_stopping(val_acc[num_target]['abs_loss'])            

        # visualize the result
        CLIreporter(targets, train_loss, train_acc, val_loss, val_acc)
        print('Epoch {}. Current learning rate {}. Took {:2.2f} sec'.format(epoch+1,optimizer.param_groups[0]['lr'],te-ts))

        # saving the checkpoint
        #if train_acc[targets[0]] > 0.9:
        checkpoint_dir = checkpoint_save(net, save_dir, epoch, val_acc, val_accs, args)

        # early stopping 
        #if early_stopping.early_stop: 
        #    break

    # test
    net.to('cpu')
    torch.cuda.empty_cache()

    net = checkpoint_load(net, checkpoint_dir)

    # setting DataParallel
    devices = []
    for d in range(torch.cuda.device_count()):
        devices.append(d)
    net = nn.DataParallel(net, device_ids = devices)
    net.cuda()

    if args.model.find('flipout_') != -1: 
        test_acc, confusion_matrices, predicted_score = test(net, partition, args, num_monete_carlo=50)
    else: 
        test_acc, confusion_matrices, predicted_score = test(net, partition, args)

    # summarize results
    result = {}
    result['train_losses'] = train_losses
    result['train_accs'] = train_accs
    result['val_losses'] = val_losses
    result['val_accs'] = val_accs

    result['train_acc'] = train_acc
    result['val_acc'] = val_acc
    result['test_acc'] = test_acc 
    if args.get_predicted_score: 
        result['predicted_score'] = predicted_score  
    
    if confusion_matrices != None:
        result['confusion_matrices'] = confusion_matrices

    return vars(args), result
## ==================================== ##




if __name__ == "__main__":

    ## ========= Setting ========= ##
    args = argument_setting()
    current_dir = os.getcwd()
    image_dir, phenotype_dir = check_study_sample(study_sample=args.study_sample)
    image_files = loading_images(image_dir=image_dir, args=args, study_sample=args.study_sample)

    if args.undersampling_dataset_target: 
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample, undersampling_dataset_target=args.undersampling_dataset_target)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, undersampling_dataset_target=args.undersampling_dataset_target, study_sample=args.study_sample)
        partition = undersampling_ALLset(imageFiles_labels, target_list, undersampling_dataset_target=args.undersampling_dataset_target ,args=args)
    elif args.partitioned_dataset_number is not None: 
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample, partitioned_dataset_number=args.partitioned_dataset_number)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, partitioned_dataset_number=args.partitioned_dataset_number, study_sample=args.study_sample)
        partition = partition_dataset_predefined(imageFiles_labels, target_list, partitioned_dataset_number=args.partitioned_dataset_number, args=args)
    else:
        subject_data, target_list = loading_phenotype(phenotype_dir=phenotype_dir, args=args, study_sample=args.study_sample)
        ## data preprocesing categorical variable and numerical variables
        imageFiles_labels = combining_image_target(subject_data, image_files, target_list, study_sample=args.study_sample)
        partition = partition_dataset(imageFiles_labels, target_list, args=args)
    ## ====================================== ##


    ## ========= Run Experiment and saving result ========= ##
    # seed number
    set_random_seed(args.seed)
    save_dir = current_dir + '/result'
    
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    args.exp_name = args.exp_name + f'_{hash_key}'


    # Run Experiment
    setting, result = experiment(partition, subject_data, save_dir, deepcopy(args))

    # Save result
    save_exp_result(save_dir, setting, result)
    ## ====================================== ##
