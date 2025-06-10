import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import os
import sys

DATASET_PATH = os.getenv('DATASET_PATH')

transform_train=transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def get_train_data_loader(batch_size):
    train_set=torchvision.datasets.CIFAR10(
    root=DATASET_PATH,
    train=True,
    download=True,
    transform=transform_train    
    )
    return torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)

def get_test_data_loader(batch_size):
    test_set=torchvision.datasets.CIFAR10(
    root=DATASET_PATH,
    train=False,
    download=True,
    transform=transform_test 
    )
    return torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=True)

def get_mean_and_std(dataset):
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=True)
    mean=torch.zeros(3)
    std=torch.zeros(3)
    print('Computing mean and std')
    for image,label in dataloader:
        for i in range(3):
            mean[i]+=image[:,i,:,:].mean()
            std[i]+=image[:,i,:,:].std()
    mean=mean/len(dataset)
    std=std/len(dataset)
    return mean,std


def get_distributed_data_loaders(batch_size, rank=None, world_size=None):
    """Get data loaders with distributed sampling if rank and world_size are provided."""
    if rank is not None and world_size is not None:
        # Distributed training
        train_dataset = torchvision.datasets.CIFAR10(
            root="/data/datasets/",
            train=True,
            download=True,
            transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="/data/datasets/",
            train=False,
            download=True,
            transform=transform_test
        )
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
        
        trainloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True
        )
        testloader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            sampler=test_sampler,
            num_workers=4,
            pin_memory=True
        )
        return trainloader, testloader, train_sampler
    else:
        # Single GPU training
        trainloader = get_train_data_loader(batch_size=batch_size)
        testloader = get_test_data_loader(batch_size=batch_size)
        return trainloader, testloader, None