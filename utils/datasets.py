from torchvision.datasets import CIFAR100, MNIST, FashionMNIST, CIFAR10
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from .utils import TinyImageNetDataset
import torch

class Cutout:
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
    
def build_dataset(args):
    train_transform, valid_transform = build_transform(args)
    if args.dataset == 'CIFAR100':
        # Load CIFAR100 dataset
        train_dataset = CIFAR100(root='Data/', train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(root='Data/', train=False, transform=valid_transform)
        
        num_classes = 100
        return (train_dataset, test_dataset, num_classes)
    elif args.dataset == 'CIFAR10':
        # Load CIFAR100 dataset
        train_dataset = CIFAR10(root='Data/', train=True, transform=train_transform, download=True)
        test_dataset = CIFAR10(root='Data/', train=False, transform=valid_transform)
       
        num_classes = 10
        return (train_dataset, test_dataset, num_classes)
    elif args.dataset == 'TinyImagenet':
        # Load TinyImagenet dataset
        train_dataset = TinyImageNetDataset(root_dir = "utils/TinyImagenet", mode='train', 
                                       preload=False, load_transform=None,transform=train_transform)

        test_dataset = TinyImageNetDataset(root_dir = "utils/TinyImagenet", mode='val', 
                                    preload=False, load_transform=None,transform= valid_transform)
        num_classes = 200
        return (train_dataset, test_dataset, num_classes)
 
    

    
def build_transform(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    train_transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.RandomCrop(args.input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(10)
    ])
    valid_transform = transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform
