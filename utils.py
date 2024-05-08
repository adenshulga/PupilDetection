import torch
import os
import json
from sklearn.model_selection import train_test_split

import cv2

import numpy as np

def train_valid_test_split(*args, test_size, valid_size,  **kwargs):
    train_valid_splits = train_test_split(*args, test_size=test_size, **kwargs)
    train_valid_arrays = train_valid_splits[::2]
    test_arrays = train_valid_splits[1::2]
    
    # Adjust valid_size to account for the initial split
    valid_size_adj = valid_size / (1 - test_size)
    
    # Split train_valid into final train and valid
    final_train_arrays = []
    final_valid_arrays = []
    for array in train_valid_arrays:
        train_split, valid_split = train_test_split(array, test_size=valid_size_adj, **kwargs)
        final_train_arrays.append(train_split)
        final_valid_arrays.append(valid_split)
    
    # Combine all the splits into a single list to return
    # This interleaves the final train, valid, and test arrays for each input array
    final_splits = []
    for train, valid, test in zip(final_train_arrays, final_valid_arrays, test_arrays):
        final_splits.extend([train, valid, test])
    
    return final_splits

    

def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including tensor, module...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


    
class Config:
    
    def __init__(self, model_name = None, dataset_name = None, seed = None) -> None:
        abs_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(abs_path) as file:
            config = json.load(file)
        
        for key in config:
            setattr(self, key, config[key])
        
        
        if seed:
            self.seed = seed

    def modify_config(self, model_name = None, dataset_name = None, seed = None) :
        abs_path = os.path.join(os.path.dirname(__file__), "config.json")
        with open(abs_path) as file:
            config = json.load(file)
        abs_path_models = os.path.join(os.path.dirname(__file__), "models_configs.json")
        with open(abs_path_models) as file:
            models_config = json.load(file)
        if model_name is None:
            model_name = config['model_name']
        if dataset_name is None:
            dataset_name = config['dataset_name']
        # print(models_config)
        model_settings = models_config[model_name][dataset_name]
        
        for key in config:
            setattr(self, key, config[key])
        
        for key in model_settings:
            setattr(self, key, model_settings[key])
        
        if seed:
            self.seed = seed

    def __str__(self) -> str:
        attributes = [f"'{key}': '{value}' \n" for key, value in self.__dict__.items()]
        return '{' + ', '.join(attributes) + '}'

config = Config()


def get_frames_seq(path: str) -> np.ndarray:

    cap = cv2.VideoCapture(path)

    frames = []

    ret = True
    while ret:
        ret, frame = cap.read()
        if ret:
            # Normalization
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

    cap.release()
    video_tensor = np.stack(frames, axis=0)

    return video_tensor

def get_video_names(path: str) -> list[str]:
    avi_files = [file.rsplit('.', 1)[0] for file in os.listdir(path) if file.endswith('.avi')]

    return avi_files


def get_gt_coord(path: str) -> np.ndarray:
    coordinates = np.genfromtxt(path)

    return coordinates



