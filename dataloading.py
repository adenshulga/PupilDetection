from torch.utils.data import Dataset, DataLoader
import torch 

import numpy as np

from utils import train_valid_test_split, get_frames_seq, get_gt_coord
from sklearn.model_selection import train_test_split
from torchvision import datasets, models, transforms

class ImageDataset(Dataset):
    def __init__(self, frames: np.ndarray,target: np.ndarray, transform=None):

        self.target = target.astype(np.float32)
        self.frames = frames.astype(np.float32)

        self.transform = transform


    def __getitem__(self, idx):

        image = self.frames[idx]
        target = self.target[idx]
        
        # преобразуем, если нужно
        if self.transform:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self):
        return len(self.target)
    

def create_dataloaders(config):

    video_path = config.path + '.avi'
    gt_path = config.path + '.txt'

    video_tensor = get_frames_seq(video_path)
    gt = np.genfromtxt(gt_path)
    train_frames, test_frames, train_gt, test_gt= train_test_split(video_tensor, gt)

    transform = None

    train_dataset = ImageDataset(train_frames,
                                 train_gt,
                                 transform=transform)
    
    valid_dataset = ImageDataset(test_frames,
                                test_gt,
                                transform)
    

    train_loader = DataLoader(dataset=train_dataset,
                                            batch_size=config.batch_size,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=config.num_workers)
    
    valid_loader = DataLoader(dataset=valid_dataset,
                                            batch_size=config.batch_size,
                                            pin_memory=True,
                                            num_workers=config.num_workers)


    return train_loader, valid_loader, None

def load_test(config):
    video_path = config.path + '.avi'
    gt_path = config.path + '.txt'

    video_tensor = get_frames_seq(video_path)
    gt = np.genfromtxt(gt_path)

    transform = None

    test_dataset = ImageDataset(video_tensor,
                                gt,
                                transform)
    

    test_loader = DataLoader(dataset=test_dataset,
                                        batch_size=config.batch_size,
                                        pin_memory=True,
                                        num_workers=config.num_workers)
    
    return test_loader
