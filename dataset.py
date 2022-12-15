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
    def __init__(self, dataset_root, train, transform):
        super().__init__()
        self.samples, self.gt = load_image(dataset_root, train)
        self.transform_fn = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.samples[index]).convert('RGB') # pixel image

        '''
        if len(list(image.mode)) > 3:
            image =
        '''

        image = self.transform_fn(image)


        gt = self.gt[index]
        ret = {
            'x': image,
            'y': torch.LongTensor([gt]).squeeze()
        }

        return ret



if __name__=='__main__':
    import numpy as np
    import cv2

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), # normalisation and tensor
    ])


    dataset = TomatoDiseaseDataset('./archive/', True, transform) #for training

    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1)
    for data in dataloader:
        image = data['x']
        print(image.shape)
        break

    '''
    for data in dataset:
        image = data['x']
        image = image * 255.0 # [0,1] -> [0,255]
        image = image.numpy().astype(np.uint8) # dtype -> ndarray
        # c h w -> h w c
        image = image.transpose(1,2,0)
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('test.png', image)



        break
    '''









#
