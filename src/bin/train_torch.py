import sys
import argparse
import os
import time
from tqdm import tqdm
import torch
import pkbar
import pandas as pd
from torch import nn
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader

if './' not in sys.path:
    sys.path.append('./')

from src.utils.config import get_config
from src.generators.torch_gen import CelebAGenerator
from src.models.celeb_a_torch import CelebAModel


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('config', help='The path to the config file')
    parser.add_argument('--finetune', action='store_true')
    return parser.parse_args(args)


def create_train_directory(save_folder_root):
    # Create new folder with current date
    date = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    save_folder = os.path.join(save_folder_root, date)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def _get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _optimizer(model, conf, criterion, params_to_update, finetune):
    lr = conf.train.lr
    wd = 1e-4 if finetune else 0
    if conf.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_to_update,
                                    lr=lr,
                                    momentum=0.9,
                                    weight_decay=wd)
    elif conf.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_to_update, lr=lr, weight_decay=wd)
    return optimizer


def _recall(labels, preds):
    recall = torch.where(labels > 0.5, preds, torch.zeros_like(preds))
    recall = torch.sum(recall) / torch.sum(labels)
    return recall


def _precision(labels, preds):
    precision = torch.where(preds > 0.5, labels, torch.zeros_like(labels))
    precision = torch.sum(precision) / torch.sum(preds)
    return precision

def _accuracy(labels, preds):
    equality = torch.sum(labels == preds)
    accuracy = equality / labels.nelement()
    return accuracy

def _init_metrics():
    metrics = ['loss', 'recall', 'precision', 'accuracy']
    metrics = {k: 0. for k in metrics}
    return metrics


def _batch_metrics(labels, preds, loss):
    data = {}
    data['loss'] = loss.item()
    data['recall'] = _recall(labels, preds)
    data['precision'] = _precision(labels, preds)
    data['accuracy'] = _accuracy(labels, preds)
    return data


def _update_running_metrics(orig, new):
    for k, v in new.items():
        orig[k] += new[k]
    return orig

def train_multilabel(datasets, save_folder, conf, finetune=False):

    val_acc_history = []
    stats = pd.DataFrame()
    device = _get_device()
    save_prefix = conf.train.optimizer

    # Define model
    model = CelebAModel(conf.train.n_features)
    if not finetune:
        print('Loading in finetuned network..')
        model.load_state_dict(torch.load(conf.path.finetuned))
        print('Done.')
    model.train().to(device)

    criterion = torch.nn.BCELoss()
    updatable_params = model.freeze() if finetune else model.parameters()
    optimizer = _optimizer(model, conf, criterion, updatable_params, finetune)

    num_epochs = 1 if finetune else conf.train.epochs

    for epoch in range(num_epochs):

        epoch_data = {}

        for phase in ['train', 'val']:

            running_metrics = _init_metrics()
            shuffle = phase=='train'
            data_loader = DataLoader(datasets[phase],
                batch_size=conf.train.batch_size,
                shuffle=shuffle,
                num_workers=4,
                prefetch_factor=5,
                pin_memory=True)
            n_iter = np.ceil(len(data_loader.dataset) / data_loader.batch_size)
            
            kbar = pkbar.Kbar(target=n_iter,
                              epoch=epoch,
                              num_epochs=num_epochs,
                              width=8,
                              always_stateful=False)

            # Iterate over data.
            for i, (inputs, labels) in enumerate(data_loader):
                # Data from generator
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Make gradients zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # Get loss & prediction
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Metrics
                    preds = torch.where(outputs > 0.5, 1, 0)
                    new_metrics = _batch_metrics(labels, preds, loss)

                    # Backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update loss & acc for epoch end
                running_metrics = _update_running_metrics(
                    running_metrics, new_metrics)

                # Update progress bar
                new_values = [(phase + '_' + name, value)
                              for (name, value) in new_metrics.items()]
                kbar.update(i, values=new_values)

            for name, value in running_metrics.items():
                if name != 'loss':
                    value = value.detach().cpu().numpy()
                epoch_data[phase+'_'+name] = value / n_iter
            
        stats = stats.append(epoch_data, ignore_index=True)
        save_path = os.path.join(save_folder, 'training_log.csv')
        stats.to_csv(save_path, index=False)

        if not finetune:
            cond1 = epoch < 10
            cond2 = (epoch + 1) % 10 == 0
            if cond1 or cond2:
                filename = save_prefix + str(epoch + 1) + '.pt'
                path = os.path.join(save_folder, filename)
                torch.save(model.state_dict(), path)

        print()

    if finetune:
        path = os.path.join(save_folder, 'incetionv1_finetuned.pt')
        torch.save(model.state_dict(), path)


def run(conf, finetune):

    # Create generators
    datasets = {
        x: CelebAGenerator.from_conf(conf, is_train=x == 'train')
        for x in ['train', 'val']
    }

    # Get new training directory and save weights before training
    save_folder = create_train_directory(conf.path.models_root)

    train_multilabel(datasets, save_folder, conf, finetune=finetune)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    conf = get_config(args.config)
    run(conf, args.finetune)


if __name__ == '__main__':
    main()