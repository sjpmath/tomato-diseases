import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

CATEGORY_TO_INDEX = {
    'Bacterial_spot': 0,
    'Early_blight': 1,
    'healthy': 2,
    'Late_blight': 3,
    'Leaf_Mold': 4,
    'powdery_mildew': 5,
    'Septoria_leaf_spot': 6,
    'Spider_mites Two-spotted_spider_mite': 7,
    'Target_Spot': 8,
    'Tomato_mosaic_virus': 9,
    'Tomato_Yellow_Leaf_Curl_Virus': 10
}

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
        if category_name == ".DS_Store": continue
        category_dir = os.path.join(image_dir, category_name)
        for image_name in os.listdir(category_dir):
            if not image_name.split('.')[-1] in ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']:
                print(image_name)
                continue
            image_path = os.path.join(category_dir, image_name)
            samples.append(image_path)
            gt.append(CATEGORY_TO_INDEX[category_name])
    return samples, gt

class TomatoDiseaseDataset(Dataset):
    def __init__(self, dataset_root, train):
        super().__init__()
        self.samples, self.gt = load_image(dataset_root, train)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.samples[index]) # pixel image
        gt = self.gt[index]
        ret = {
            'x': image,
            'y': gt
        }

        return ret



if __name__=='__main__':
    dataset = TomatoDiseaseDataset('./archive/', True) #for training
    for data in dataset:
        print(data['x'].size, data['y'])
