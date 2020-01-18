

import torch
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from skimage import io,transform


from dataset import CatDogDataset
from model import *


class Trainer():
    def __init__(self,model,path='data/Cat_Dog_data/'):
        self.device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image_size = (64, 64)
        self.image_row_size = self.image_size[0] * self.image_size[1] * 3

        self.train_transform = transforms.Compose([
                                transforms.RandomRotation(30),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.Resize(self.image_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.test_transform = transforms.Compose([transforms.Resize(self.image_size), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.path = path 
        self.criterion = nn.CrossEntropyLoss()
        

        self.net = model
        self.net.to(self.device)

    def get_loader(self,path,batch_size = 64):
        train_data = CatDogDataset(path+'train/',  transform=self.train_transform)
        test_data = CatDogDataset(path+'test/',  transform=self.test_transform)

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=4)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=False, num_workers=4)


        return trainloader,testloader

    
    def train(self,epochs=5,lr=0.001,batch_size=64):
        train_loader,_ = self.get_loader(self.path,batch_size=batch_size)


        optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self.net.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data,target = data.to(self.device),target.to(self.device)
                
                optimizer.zero_grad()
                output = self.net(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 50 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item()))
            
    def test(self):
        _,test_loader = self.get_loader(self.path)
        self.net.eval()

        accuracy_list = []
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            output = self.net(data)
            test_loss += self.criterion(output, target).item()                                                              
            pred = output.data.max(1, keepdim=True)[1]                                                                
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        accuracy_list.append(accuracy)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),accuracy))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_dir',default='data/Cat_Dog_data/',help='FILE DIR')
    parser.add_argument('-batch_size',default=64,help='BATCH SIZE')
    parser.add_argument('-lr',default=0.001, help='LEARNING RATE')
    parser.add_argument('-epoch',default=5, help='EPOCH')
    parser.add_argument('-model',default='SIMPLE',choices=['SIMPLE', 'DEEPER'],
                    help='Choose the model you are interested in')
    args = parser.parse_args()

    if args.file_dir and args.batch_size and args.lr and args.epoch:
        print('Training on Default hyperparameters')
        print('----------------------------------')

    if args.model == 'SIMPLE':
        model = CNN()
    else:
        model = CNN_ONE()
    print('Using Model With Architecture:\n')
    print(model)
    print('--------')

    trainer = Trainer(model,path=args.file_dir)
    trainer.train(epochs=args.epoch,lr=args.lr,batch_size=args.batch_size)
    trainer.test()