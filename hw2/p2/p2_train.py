import os
import sys
import time
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

import config as cfg

def plot_learning_curve(logfile_dir, result_lists):
    ################################################################
    # TODO:                                                        #
    # Plot and save the learning curves under logfile_dir, you can #
    # use plt.plot() and plt.savefig().                            #
    #                                                              #
    # NOTE:                                                        #
    # You have to attach four plots of your best model in your     #
    # report, which are:                                           #
    #   1. training accuracy                                       #
    #   2. training loss                                           #
    #   3. validation accuracy                                     #
    #   4. validation loss                                         #
    #                                                              #
    # NOTE:                                                        #
    # This function is called at end of each epoch to avoid the    #
    # plot being unsaved if early stop, so the result_lists's size #
    # is not fixed.                                                #
    ################################################################
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(result_lists['train_acc'], label='train_acc')
    plt.plot(result_lists['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(result_lists['train_loss'], label='train_loss')
    plt.plot(result_lists['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(logfile_dir, 'learning_curve.png'))
    plt.close()

def get_pseudo_labels(unlabel_loader, model, threshold=cfg.threshold_k):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)

    dataset = []

    # Iterate over the dataset by batches.
    for batch in unlabel_loader:
        img = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(device))

        # Obtain the probability distributions by applying softmax on logits.
        probs = softmax(logits)

        # Filter the data and construct a new dataset.
        for i in range(len(probs)):
            # Keep the data with a confidence value greater than the threshold.
            if torch.max(probs[i]) > threshold:
                dataset.append({
                    'images': img[i],
                    'labels': torch.argmax(probs[i]).item()
                })

    # # Turn off the eval mode.
    model.train()
    return dataset

def train(model, train_loader, val_loader, unlabel_loader, logfile_dir, model_save_dir, criterion, optimizer, scheduler, device):

    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        ##### TRAINING #####
        train_start_time = time.time()
        train_loss = 0.0
        train_correct = 0.0

        # TODO: Semi-supervised learning using unlabel data
        if epoch >= cfg.start_unlabel_epoch:
            pseudo_dataset = get_pseudo_labels(unlabel_loader, model, threshold=cfg.threshold_k)
            train_dataset = train_loader.dataset
            concat_dataset = ConcatDataset([train_dataset, pseudo_dataset])
            concat_loader = DataLoader(concat_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
            print(f'[{epoch + 1}/{cfg.epochs}] New dataset size: {len(concat_dataset)}')
        
        new_loader = concat_loader if epoch >= cfg.start_unlabel_epoch else train_loader
        model.train()
        for batch, data in enumerate(new_loader):
            sys.stdout.write(f'\r[{epoch + 1}/{cfg.epochs}] Train batch: {batch + 1} / {len(new_loader)}')
            sys.stdout.flush()
            # Data loading.
            images, labels = data['images'].to(device), data['labels'].to(device) # (batch_size, 3, 32, 32), (batch_size)
            # Forward pass. input: (batch_size, 3, 32, 32), output: (batch_size, 10)
            pred = model(images)
            # Calculate loss.
            loss = criterion(pred, labels)
            # Backprop. (update model parameters)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Evaluate.
            train_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
            train_loss += loss.item()
        # Print training result
        train_time = time.time() - train_start_time
        train_acc = train_correct / len(new_loader.dataset)
        train_loss /= len(new_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)
        print()
        print(f'[{epoch + 1}/{cfg.epochs}] {train_time:.2f} sec(s) Train Acc: {train_acc:.5f} | Train Loss: {train_loss:.5f}')

        ##### VALIDATION #####
        model.eval()
        with torch.no_grad():
            val_start_time = time.time()
            val_loss = 0.0
            val_correct = 0.0
            #############################################################
            # TODO:                                                     #
            # Finish forward part in validation, you can refer to the   #
            # training part.                                            #
            #                                                           #
            # NOTE:                                                     #
            # You don't have to update parameters, just record the      #
            # accuracy and loss.                                        #
            #############################################################

            for batch, data in enumerate(val_loader):
                images, labels = data['images'].to(device), data['labels'].to(device)
                pred = model(images)
                loss = criterion(pred, labels)
                val_correct += torch.sum(torch.argmax(pred, dim=1) == labels)
                val_loss += loss.item()
            
            ######################### TODO End ##########################

        # Print validation result
        val_time = time.time() - val_start_time
        val_acc = val_correct / len(val_loader.dataset)
        val_loss /= len(val_loader)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        print(f'[{epoch + 1}/{cfg.epochs}] {val_time:.2f} sec(s) Val Acc: {val_acc:.5f} | Val Loss: {val_loss:.5f}')
        
        # Scheduler step
        scheduler.step()

        ##### WRITE LOG #####
        is_better = val_acc >= best_acc
        epoch_time = train_time + val_time
        write_result_log(os.path.join(logfile_dir, 'result_log.txt'), epoch, epoch_time, train_acc, val_acc, train_loss, val_loss, is_better)

        ##### SAVE THE BEST MODEL #####
        if is_better:
            print(f'[{epoch + 1}/{cfg.epochs}] Save best model to {model_save_dir} ...')
            torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_best.pth'))
            best_acc = val_acc

        ##### PLOT LEARNING CURVE #####
        ##### TODO: check plot_learning_curve() in this file #####
        current_result_lists = {
            'train_acc': train_acc_list,
            'train_loss': train_loss_list,
            'val_acc': val_acc_list,
            'val_loss': val_loss_list
        }
        plot_learning_curve(logfile_dir, current_result_lists)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dataset directory', type=str, default='../hw2_data/p2_data/')
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    # Experiment name
    exp_name = cfg.model_type + datetime.now().strftime('_%Y_%m_%d_%H_%M_%S') + '_' + cfg.exp_name

    # Write log file for config
    logfile_dir = os.path.join('./experiment', exp_name, 'log')
    os.makedirs(logfile_dir, exist_ok=True)
    write_config_log(os.path.join(logfile_dir, 'config_log.txt'))

    # Fix a random seed for reproducibility
    set_seed(9527)

    # Check if GPU is available, otherwise CPU is used
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Device:', device)

    ##### MODEL #####
    ##### TODO: check model.py #####
    model_save_dir = os.path.join('./experiment', exp_name, 'model')
    os.makedirs(model_save_dir, exist_ok=True)

    if cfg.model_type == 'mynet':
        model = MyNet()
    elif cfg.model_type == 'resnet18':
        model = ResNet18()
    else:
        raise NameError('Unknown model type')

    model.to(device)

    ##### DATALOADER #####
    ##### TODO: check dataset.py #####
    train_loader = get_dataloader(os.path.join(dataset_dir, 'train'), batch_size=cfg.batch_size, split='train')
    val_loader   = get_dataloader(os.path.join(dataset_dir, 'val'), batch_size=cfg.batch_size, split='val')
    unlabel_loader = get_dataloader(os.path.join(dataset_dir, 'unlabel'), batch_size=cfg.batch_size, split='unlabel')

    ##### LOSS & OPTIMIZER #####
    criterion = nn.CrossEntropyLoss()
    if cfg.use_adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=0.1)
    
    ##### TRAINING & VALIDATION #####
    ##### TODO: check train() in this file #####
    train(model          = model,
          train_loader   = train_loader,
          val_loader     = val_loader,
          unlabel_loader = unlabel_loader,
          logfile_dir    = logfile_dir,
          model_save_dir = model_save_dir,
          criterion      = criterion,
          optimizer      = optimizer,
          scheduler      = scheduler,
          device         = device)
    
if __name__ == '__main__':
    main()
