import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
# from Mydataset import get_train_data_loader,get_test_data_loader
# from ResNet18 import ResNet18
from utils.ResNet import ResNet18
from tqdm import tqdm
import os
# from torch.utils.tensorboard import SummaryWriter

#Preseting parmaters
SAVE_PATH="/home/cx922/AICrossSim/models/train_2/"
DATASET_PATH="/home/cx922/AICrossSim/data/"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

TRAINSET_LENGTH=50000
TESTSET_LENGTH=10000
#Hyperparmeters:
num_epochs=300
learning_rate=0.1
batch_size=256
device= 'cuda' if torch.cuda.is_available() else 'cpu'

import torch
import torchvision
import torchvision.transforms as transforms


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


def test(network,testloader):
    network.eval()
    total_correct=0
    total_loss=0
    with torch.no_grad():
        for images,labels in testloader:
            images=images.to(device)
            labels=labels.to(device)
            preds=network(images)
            loss=F.cross_entropy(preds,labels)
            total_loss+=loss.item()
            total_correct+=preds.argmax(dim=1).eq(labels).sum().item()
    network.train()
    return total_loss,total_correct/TESTSET_LENGTH

'''采用边训练，边测试的流程，保存测试集最高准去率模型'''

def train():
    #1.initilazing network
    network=ResNet18()
    network=network.to(device)
    print("Initializing Network")
    #2.prepare data
    trainloader=get_train_data_loader(batch_size=batch_size)
    testloader=get_test_data_loader(batch_size=batch_size)
    #3.prepare optimizer
    optimizer=optim.Adam(network.parameters(),lr=learning_rate)
    #optimizer=optim.SGD(network.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    #可选项：衰减学习率
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[150,250,300],gamma=0.1)#每10次训练衰减10倍
    #4.initilazing tensorboard
    comment=f'ResNet18 batch_size={batch_size} lr={learning_rate} device={device}'
    # tb=SummaryWriter(comment=comment)
    best_acc=0.0
    for epoch in tqdm(range(num_epochs)):
        train_loss=0
        train_correct=0
        for images,labels in trainloader:
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            preds=network(images)
            loss=F.cross_entropy(preds,labels)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_correct+=preds.argmax(dim=1).eq(labels).sum().item()
            
        
        present_trainset_acc=train_correct/TRAINSET_LENGTH
        test_loss,present_testset_acc=test(network,testloader)
        

        tqdm.write(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {present_trainset_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {present_testset_acc:.4f}')
        scheduler.step()
        '''只保存测试集准确率不断上升的epoch'''
        if present_testset_acc>best_acc:
            state={
                'network':network.state_dict(),
                'accuracy':present_testset_acc,
                'optimizer':optimizer.state_dict(),
                'epoch':epoch
            }
            torch.save(state,os.path.join(SAVE_PATH, "model_best.pkl"))
            best_acc=present_testset_acc
        print("epoch",epoch,"loss",train_loss,"Train acc",present_trainset_acc,"Test acc ",present_testset_acc)
    # tb.close()

if __name__=='__main__':
    train()