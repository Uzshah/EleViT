from utils import *
from models import *
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
import os
import random
import string
import shutil


def generate_random_name(length=4):
    syllables = ['ba', 'do', 'fi', 'ja', 'lo', 'ma', 'ne', 'pi', 'ro', 'ta']
    random_name = ''.join(random.choice(syllables) for _ in range(length))
    return random_name




def create_folder(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        try:
            # Create the folder
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            print(f"Error creating folder '{folder_path}': {e}")
    else:
        print(f"Folder '{folder_path}' already exists.")


def get_args_parser():
    parser = argparse.ArgumentParser(
        'EfficientFormer training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=150, type=int)

    # Model parameters
    parser.add_argument('--model', default='EleViT_L', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')


    # Optimizer parameters
    parser.add_argument('--opt', default='AdamW', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='CosineAnnealingLR', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 2e-3)')


    # * CutMix params
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')


    # Dataset parameters
    parser.add_argument('--dataset', default='CIFAR100', choices=['CIFAR100', 'CIFAR10', 'TinyImagenet'],
                        type=str, help='Image Net dataset path')
    
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--output_dir', default='check_points',
                        help='path where to save, empty for no saving')
    parser.add_argument('--eval', default='',
                        help='path from where to load')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    return parser
    
def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # Fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    train_dataset, test_dataset, args.num_classes = build_dataset(args)
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    sampler_test = torch.utils.data.DistributedSampler(
            test_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    ## Model calling
    model = eval(args.model)(num_classes = args.num_classes)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    #print(f'scheduler {scheduler}')
    ## Loss
    if args.cutmix:
        # Create data loaders for training and validation
        train_loader = DataLoader(CutMix(train_dataset, args.num_classes,
                       beta=1.0, prob=0.5, num_mix=2), batch_size=args.batch_size, 
                       sampler=sampler_train, num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
         sampler=sampler_test, num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,)
        criterion = CutMixCrossEntropyLoss(True).to(device)
    else:
        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
         sampler=sampler_train, num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
         sampler=sampler_test, num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,)
        criterion = CrossEntropyLoss()
        
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu])
        model_without_ddp = model.module   
        
    ## optimizer calling
    optimizer = eval(args.opt)(model_without_ddp.parameters(), lr = args.lr)
    
    ## schedular
    scheduler = eval(args.sched)(optimizer, T_max = args.epochs)     
    ## Model evaluation     
    if args.eval:
        eval_check_point = torch.load(args.eval)
        model.load_state_dict(eval_check_point['model'])
        valid_loss, valid_accuracy, valid_time = evalute(model, criterion, 
                                                     test_loader, device, 
                                                     args)
        print(f"validation loss: {valid_loss:.4f}, validation accuracy: {valid_accuracy:.2f}, Total time: {valid_time:.2f}")
        return 
    create_folder(args.output_dir)
    create_folder(args.output_dir+'/'+args.dataset)
    ## Create Model specific folder
    checkpoint = args.output_dir+'/'+args.dataset+'/'+args.model
    create_folder(checkpoint)
    if any(os.listdir(checkpoint)):
        source_folder = checkpoint
        backup_folder = f"{checkpoint}_backup_{generate_random_name()}"

        # Create a backup of the source folder
        shutil.copytree(source_folder, backup_folder)
        print(f"Backup is saved in {backup_folder}")
    ## Save model 
    torch.save(model, f'{checkpoint}/{args.model}.pth')
    
    ## Initialize some variable
    best_valid_acc = 0.0
    best_weights = None
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    if args.resume:
        check_point = torch.load(args.resume)
        model.load_state_dict(check_point['model'])
        if 'optimizer' in check_point and 'scheduler' in check_point and 'epoch' in check_point:
            optimizer.load_state_dict(check_point['optimizer'])
            scheduler.load_state_dict(check_point['scheduler'])
            args.start_epoch = check_point['epoch'] + 1
        
    print(f"Training Start from epochs {args.start_epoch}....\n")
    
    file = open(f"{checkpoint}/log.txt", "w") 

    for epoch in range(args.start_epoch, args.epochs):
        train_loss, train_accuracy, epoch_time = train_one_epoch(model, criterion,
                                                            optimizer, scheduler,
                                                            train_loader, device,
                                                            args)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        print()
        text = f'Epoch [{epoch+1}/{args.epochs}], training loss: {train_loss:.4f}, training accuracy: {train_accuracy:.2f}, Total time: {epoch_time:.2f}'
        print(text)
        file.write(text)
        file.write('\n')
        
        
        ## Validate model perform over the test
        valid_loss, valid_accuracy, valid_time = evalute(model, criterion, 
                                                     test_loader, device, 
                                                     args)
        valid_losses.append(valid_loss)
        valid_acc.append(valid_accuracy)
        text = f'validation loss: {valid_loss:.4f}, validation accuracy: {valid_accuracy:.2f}, Total time: {valid_time:.2f}'
        print(text)
        file.write(text)
        file.write('\n')
        model_checkpoint_path = f'{checkpoint}/last.pth'
        torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': args,},
                model_checkpoint_path)
        ## check model performance if model improve and write new checkpoint
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            best_weights = model.state_dict()
            # After training for an epoch, save the model checkpoint
            model_checkpoint_path = f'{checkpoint}/{args.model}_state.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'args': args,},
                model_checkpoint_path)
            print(f"Model checkpoint saved for epoch {epoch + 1} at {model_checkpoint_path}")
            
    print(f"Best validation accuracy is {best_valid_acc}")
    file.close()
    print(f"Every epoch data is recorded in the log.txt file {checkpoint}")
    print("Model training is completed Successfully")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'EleViT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
