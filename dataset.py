import os
import glob
from PIL import Image
from torchvision import transforms
# from skimage import io,transform
from torch.utils.data import DataLoader,Dataset

class CustomDataset(Dataset):
    def __init__(self,path,transform=None):
        self.classes = os.listdir(path)
        self.classes = [i for i in self.classes if not i.startswith('.')]
        self.file_list = [os.listdir(path+'/'+i) for i in self.classes]

        self.transform = transform

        files = []
        for i, className in enumerate(self.classes):
            for fileName in self.file_list[i]:
                files.append([i, className, path+'/'+className+'/'+fileName])
        self.file_list = files
        files = None

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fileName = self.file_list[idx][2]
        classCategory = self.file_list[idx][0]
        im = Image.open(fileName)
        if self.transform:
            im = self.transform(im)
        return im, classCategory
        
        
