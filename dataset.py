import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def load_image(dataset_root, train):
    '''
        args:
            dataset_root (str): path to the dataset
            train (bool): train if true, valid else
        return:
            samples (List[str]): path to all the images
            gt (List[int]): ground truth
    '''
    samples = []
    gt = []
    if train:
        image_dir = os.path.join(dataset_root, 'train')
    else:
        image_dir = os.path.join(dataset_root, 'valid')
    for category_name in os.listdir(image_dir):
        print(category_name)

class TomatoDiseaseDataset(Dataset):
    def __init__(self, dataset_root, train):
        super().__init__()
        load_image(dataset_root, train)


    def __len__(self):
        pass

    def __getitem__(self):
        pass


if __name__=='__main__':
    dataset = TomatoDiseaseDataset('./archive/', True) #for training
