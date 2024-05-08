from dataloading import ImageDataset
from training_routine import evaluate, train_model
from dataloading import create_dataloaders, load_test
from model import create_model
from utils import config, convert_to_gpu
import torch
import torch.optim as optim
import torch.nn as nn

def main(config):

    config.device = torch.device(f'cuda:0')
    print('[Info] parameters: {}'.format(config))

    model = create_model(config)
    model = convert_to_gpu(model, device =config.device)

    criterion = nn.MSELoss()
    

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)


    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=config.scheduler_step, 
                                          gamma=config.gamma)

    trainloader, devloader, _ = create_dataloaders(config)

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('[Info] Number of parameters: {}'.format(num_params))

    best_model, train_loss, valid_loss = train_model(model, 
                             criterion,
                             optimizer,
                             scheduler,
                             trainloader, 
                             devloader, 
                             config)
    
    model.load_state_dict(best_model.state_dict())
    model.eval()
    # save the model
    model_save_path = f"saved_models/my_model"
    torch.save(model.state_dict(), model_save_path)

    del trainloader, devloader

    testloader = load_test(config)

    score = evaluate(model, testloader, config)

    print(f'Score: {score}')

main(config)



    













