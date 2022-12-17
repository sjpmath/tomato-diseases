import argparse
import copy

import torch
from model import LeNet5
from dataset import TomatoDiseaseDataset
from runner import test, train
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser.parse_args()

def main():
    args = opt()
    # batched dataset
    import numpy as np
    import cv2

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(), # normalisation and tensor
    ])

    train_dataset = TomatoDiseaseDataset('./archive/', True, transform) #for training
    valid_dataset = TomatoDiseaseDataset('./archive/', False, transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    # model object
    model = LeNet5(11)

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = None
    loss_fn = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0

    TEST_EVERY = 5
    for ep in range(1,args.max_epoch):
        train(ep, args.max_epoch, model, train_dataloader, loss_fn, optimizer, writer)
        if scheduler is not None:
            scheduler.step()

        if ep%TEST_EVERY ==0:
            accuracy = test(ep, args.max_epoch, model, valid_dataloader, writer)
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                best_model = copy.deepcopy(model)


    save_path = './best.pth'
    torch.save({
        'weight' : best_model.state_dict()
    })
    print('Train finished')
    print('Best accuracy: ', best_accuracy)
    print('Your model was saved in ', save_path)


if __name__ == '__main__':
    main()
