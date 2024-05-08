
import torch
import tqdm
from copy import deepcopy
import torch.nn as nn
from tqdm import tqdm
from utils import convert_to_gpu


import numpy as np
from sklearn.metrics import mean_squared_error


def train_epoch(
        model, 
        criterion,
        optimizer, 
        scheduler, 
        trainloader,
        config) -> float:
    
    model.train()

    train_loss = 0

    for batch in tqdm(trainloader):

        ###### model specific code

        imgs, target = batch

        imgs, target = convert_to_gpu(imgs, target, device=config.device)
        imgs = imgs.permute(0, 3, 1, 2)

        prediction, _ = model(imgs)

        optimizer.zero_grad()

        loss = criterion(prediction, target)
        loss.backward()

        train_loss += loss
        optimizer.step()

    return loss

def eval_epoch(model, validloader, config) -> float:

    model.eval()
    predictions = []
    gt = []

    with torch.no_grad():
        for batch in tqdm(validloader):
            imgs, target = batch

            imgs, target = convert_to_gpu(imgs, target, device=config.device)
            imgs = imgs.permute(0, 3, 1, 2)


            ##### model specific code:
            prediction, _ = model(imgs)

            predictions.append(prediction)
            gt.append(target)

        true = torch.cat(gt, dim=0)
        pred = torch.cat(predictions, dim=0)

    return mean_squared_error(pred.cpu(), true.cpu(), squared=False)        
    
def evaluate(model, validloader, config) -> float:

    model.eval()
    predictions = []
    gt = []

    with torch.no_grad():
        for batch in tqdm(validloader):
            imgs, target = batch

            imgs, target = convert_to_gpu(imgs, target, device=config.device)
            imgs = imgs.permute(0, 3, 1, 2)


            ##### model specific code:
            prediction, x_before_flatten = model(imgs)

            predictions.append(x_before_flatten)
            gt.append(target)

        true = torch.cat(gt, dim=0)
        pred = torch.cat(predictions, dim=0)

        np.save('new_features.npy', pred.detach().cpu().numpy())
        np.save('ground_truth.npy', true.detach().cpu().numpy())


    return 1       

def train_model(
        model, 
        criterion,
        optimizer, 
        scheduler, 
        trainloader,
        validloader,
        config
        ) -> tuple[nn.Module, list, list]:
    
    train_loss_arr = []
    valid_metric_arr = []
    best_metric = np.inf
    best_model = deepcopy(model.state_dict())

    for epoch in tqdm(range(config.epochs)):
        print('[ Epoch', epoch, ']')
        train_loss = train_epoch(model, 
                                 criterion, 
                                 optimizer,
                                 scheduler,
                                 trainloader,
                                 config)
        train_loss_arr.append(train_loss)
        valid_metric = eval_epoch(model, validloader, config)
        valid_metric_arr.append(valid_metric)


        if valid_metric < best_metric:
            best_metric = valid_metric
            best_model = deepcopy(model)

    

        scheduler.step()
        print(f'Loss: {train_loss}')
        print(f'Validation score: {valid_metric}')

    return best_model, train_loss_arr, valid_metric_arr

