import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
import argparse
# from Mydataset import get_train_data_loader,get_test_data_loader
# from ResNet18 import ResNet18
from models import ResNet18
from datasets import get_train_data_loader,get_test_data_loader,get_distributed_data_loaders
from tqdm import tqdm
import os
# from torch.utils.tensorboard import SummaryWriter

#Preseting parmaters

TRAINSET_LENGTH=50000
TESTSET_LENGTH=10000
#Hyperparmeters:
device= 'cuda' if torch.cuda.is_available() else 'cpu'

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="ResNet Training, Testing, and Fine-tuning")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "finetune"],
        required=True,
        help="Mode to run: train, test, or finetune"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/models/cx922/resnet_eval/model_original/model_best.pkl",
        help="Path to the model file for testing or fine-tuning"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/data/models/cx922/resnet_eval/model_original/",
        help="Path to save the trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training and testing"
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training"
    )
    return parser.parse_args()


def test(network, testloader, rank=None):
    network.eval()
    total_correct = 0
    total_loss = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            total_loss += loss.item()
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()
            total_samples += labels.size(0)
    
    if rank is not None:
        # Gather results from all processes
        total_correct_tensor = torch.tensor(total_correct, device=device)
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)
        
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        
        total_correct = total_correct_tensor.item()
        total_loss = total_loss_tensor.item()
        total_samples = total_samples_tensor.item()
    
    network.train()
    return total_loss, total_correct / total_samples if total_samples > 0 else 0

'''采用边训练，边测试的流程，保存测试集最高准去率模型'''

def train_distributed(rank, world_size, args):
    """Distributed training function."""
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Initialize network
    network = ResNet18()
    network = network.to(device)
    network = DDP(network, device_ids=[rank])
    
    if rank == 0:
        print("Initializing Distributed Network")
    
    # Prepare data with distributed sampling
    trainloader, testloader, train_sampler = get_distributed_data_loaders(
        args.batch_size, rank, world_size
    )
    
    # Prepare optimizer
    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 300], gamma=0.1)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        # Only show progress bar on rank 0
        if rank == 0:
            pbar = tqdm(trainloader, desc=f"Training Epoch {epoch}")
        else:
            pbar = trainloader
            
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += preds.argmax(dim=1).eq(labels).sum().item()
            train_samples += labels.size(0)
        
        # Gather training metrics from all processes
        train_loss_tensor = torch.tensor(train_loss, device=device)
        train_correct_tensor = torch.tensor(train_correct, device=device)
        train_samples_tensor = torch.tensor(train_samples, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_samples_tensor, op=dist.ReduceOp.SUM)
        
        present_trainset_acc = train_correct_tensor.item() / train_samples_tensor.item()
        test_loss, present_testset_acc = test(network, testloader, rank)
        
        if rank == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss_tensor.item():.4f}, Train Acc: {present_trainset_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {present_testset_acc:.4f}")
        
        scheduler.step()
        
        # Save best model (only on rank 0)
        if rank == 0:
            SAVE_PATH = args.save_path
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            if present_testset_acc > best_acc:
                state = {
                    'network': network.module.state_dict(),  # Use .module for DDP
                    'accuracy': present_testset_acc,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(SAVE_PATH, "model_best.pkl"))
                best_acc = present_testset_acc
    
    cleanup()

def train(args):
    if args.distributed:
        mp.spawn(train_distributed, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        # Single GPU training (original code)
        network = ResNet18()
        network = network.to(device)
        print("Initializing Network")
        
        trainloader = get_train_data_loader(batch_size=args.batch_size)
        testloader = get_test_data_loader(batch_size=args.batch_size)
        
        optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 300], gamma=0.1)
        best_acc = 0.0
        
        for epoch in tqdm(range(args.epochs), desc="Training"):
            train_loss = 0
            train_correct = 0
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                preds = network(images)
                loss = F.cross_entropy(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += preds.argmax(dim=1).eq(labels).sum().item()
            
            present_trainset_acc = train_correct / TRAINSET_LENGTH
            test_loss, present_testset_acc = test(network, testloader)
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {present_trainset_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {present_testset_acc:.4f}")
            scheduler.step()
            
            SAVE_PATH = args.save_path
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            if present_testset_acc > best_acc:
                state = {
                    'network': network.state_dict(),
                    'accuracy': present_testset_acc,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(SAVE_PATH, "model_best.pkl"))
                best_acc = present_testset_acc
            print("epoch", epoch, "loss", train_loss, "Train acc", present_trainset_acc, "Test acc ", present_testset_acc)

def test_model(args):
    network = ResNet18()
    network.load_state_dict(torch.load(args.model_path)['network'])
    network = network.to(device)
    testloader = get_test_data_loader(batch_size=args.batch_size)
    test_loss, test_acc = test(network, testloader)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

def finetune_distributed(rank, world_size, args):
    """Distributed fine-tuning function."""
    setup(rank, world_size)
    
    # Set device for this process
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Load pre-trained model
    network = ResNet18()
    network.load_state_dict(torch.load(args.model_path)['network'])
    network = network.to(device)
    network = DDP(network, device_ids=[rank])
    
    if rank == 0:
        print("Loading pre-trained model for distributed fine-tuning")
    
    # Prepare data with distributed sampling
    trainloader, testloader, train_sampler = get_distributed_data_loaders(
        args.batch_size, rank, world_size
    )
    
    # Use lower learning rate for fine-tuning
    finetune_lr = args.learning_rate * 0.1
    optimizer = optim.Adam(network.parameters(), lr=finetune_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
    
    best_acc = 0.0
    finetune_epochs = min(args.epochs, 100)
    
    for epoch in range(finetune_epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss = 0
        train_correct = 0
        train_samples = 0
        
        if rank == 0:
            pbar = tqdm(trainloader, desc=f"Fine-tuning Epoch {epoch}")
        else:
            pbar = trainloader
            
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += preds.argmax(dim=1).eq(labels).sum().item()
            train_samples += labels.size(0)
        
        # Gather training metrics from all processes
        train_loss_tensor = torch.tensor(train_loss, device=device)
        train_correct_tensor = torch.tensor(train_correct, device=device)
        train_samples_tensor = torch.tensor(train_samples, device=device)
        
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_samples_tensor, op=dist.ReduceOp.SUM)
        
        present_trainset_acc = train_correct_tensor.item() / train_samples_tensor.item()
        test_loss, present_testset_acc = test(network, testloader, rank)
        
        if rank == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss_tensor.item():.4f}, Train Acc: {present_trainset_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {present_testset_acc:.4f}")
        
        scheduler.step()
        
        # Save best model (only on rank 0)
        if rank == 0:
            SAVE_PATH = args.save_path
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            if present_testset_acc > best_acc:
                state = {
                    'network': network.module.state_dict(),
                    'accuracy': present_testset_acc,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(SAVE_PATH, "model_finetuned.pkl"))
                best_acc = present_testset_acc
    
    cleanup()

def finetune(args):
    if args.distributed:
        mp.spawn(finetune_distributed, args=(args.world_size, args), nprocs=args.world_size, join=True)
    else:
        # Single GPU fine-tuning (original code)
        network = ResNet18()
        network.load_state_dict(torch.load(args.model_path)['network'])
        network = network.to(device)
        print("Loading pre-trained model for fine-tuning")
        
        trainloader = get_train_data_loader(batch_size=args.batch_size)
        testloader = get_test_data_loader(batch_size=args.batch_size)
        
        finetune_lr = args.learning_rate * 0.1
        optimizer = optim.Adam(network.parameters(), lr=finetune_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
        
        best_acc = 0.0
        finetune_epochs = min(args.epochs, 100)
        
        pbar = tqdm(range(finetune_epochs), desc="Fine-tuning")
        for epoch in pbar:
            train_loss = 0
            train_correct = 0
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                preds = network(images)
                loss = F.cross_entropy(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_correct += preds.argmax(dim=1).eq(labels).sum().item()
            
            present_trainset_acc = train_correct / TRAINSET_LENGTH
            test_loss, present_testset_acc = test(network, testloader)
            
            pbar.set_postfix({
                'Epoch': epoch,
                'Train Loss': f'{train_loss:.4f}',
                'Train Acc': f'{present_trainset_acc:.4f}',
                'Test Loss': f'{test_loss:.4f}',
                'Test Acc': f'{present_testset_acc:.4f}'
            })
            scheduler.step()
            
            SAVE_PATH = args.save_path
            if not os.path.exists(SAVE_PATH):
                os.makedirs(SAVE_PATH)
            if present_testset_acc > best_acc:
                state = {
                    'network': network.state_dict(),
                    'accuracy': present_testset_acc,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(state, os.path.join(SAVE_PATH, "model_finetuned.pkl"))
                best_acc = present_testset_acc

if __name__=='__main__':
    args = parse_args()
    
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test_model(args)
    elif args.mode == "finetune":
        finetune(args)