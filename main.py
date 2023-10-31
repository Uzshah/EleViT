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
    

    parser.add_argument('--output_dir', default='check_points',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    
    return parser
    
def main(args):
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    train_dataset, test_dataset, args.num_classes = build_dataset(args)
    
    ## Model calling
    model = eval(args.model)(num_classes = args.num_classes)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    ## optimizer calling
    optimizer = eval(args.opt)(model.parameters(), lr = args.lr)
    
    ## schedular
    scheduler = eval(args.sched)(optimizer, T_max = args.epochs)
    #print(f'scheduler {scheduler}')
    ## Loss
    if args.cutmix:
        # Create data loaders for training and validation
        train_loader = DataLoader(CutMix(train_dataset, args.num_classes,
                       beta=1.0, prob=0.5, num_mix=2), batch_size=args.batch_size, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        criterion = CutMixCrossEntropyLoss(True).to(device)
    else:
        # Create data loaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        criterion = CrossEntropyLoss()
    
    create_folder(args.output_dir)
    create_folder(args.output_dir+'/'+args.dataset)
    ## Create Model specific folder
    checkpoint = args.output_dir+'/'+args.dataset+'/'+args.model
    create_folder(checkpoint)
    
    ## Save model 
    torch.save(model, f'{checkpoint}/{args.model}.pth')
    
    ## Initialize some variable
    best_valid_acc = 0.0
    best_weights = None
    train_losses, valid_losses = [], []
    train_acc, valid_acc = [], []
    
    print("Training Start ....\n")
    
    file = open(f"{checkpoint}/log.txt", "w") 

    for epoch in range(args.epochs):
        train_loss, train_accuracy, epoch_time = train_one_epoch(model, criterion,
                                                            optimizer, scheduler,
                                                            train_loader, device,
                                                            args)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        print()
        text = f'Epoch [{epoch}/{args.epochs}], training loss: {train_loss:.4f}, training accuracy: {train_accuracy:.2f}, Total time: {epoch_time:.2f}'
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
        
        ## check model performance if model improve and write new checkpoint
        if valid_accuracy > best_valid_acc:
            best_valid_acc = valid_accuracy
            best_weights = model.state_dict()
            # After training for an epoch, save the model checkpoint
            model_checkpoint_path = f'{checkpoint}/{args.model}.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
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
