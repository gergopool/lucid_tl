import sys
import argparse
import os
import time
import copy
from tqdm import tqdm
import torch
from torch import nn
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

def _densenet():
    from torchvision.models import densenet121
    model = densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 40),
        nn.Sigmoid()
    )
    return model

def _mobilenet():
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(pretrained=True)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 40)
    model.classifier.add_module('sigmoid', nn.Sigmoid())
    return model


def _imagenetv1():
    # from lucent.modelzoo import inceptionv1
    # model = inceptionv1(pretrained=True)
    # num_ftrs = model.softmax2_pre_activation_matmul.in_features
    # model.softmax2_pre_activation_matmul = nn.Linear(num_ftrs, 40)
    model = CelebAModel()
    return model

def _optimizer(model, conf, criterion):
    params_to_update = model.parameters()
    lr = conf.train.lr
    if conf.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_to_update, lr=lr, momentum=0.9)
    elif conf.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params_to_update, lr=lr)
    return optimizer

def finetune(data_loaders, save_folder, conf):

    since = time.time()
    val_acc_history = []
    device = _get_device()

    # Define model
    # model = _mobilenet()
    # model= _densenet()
    model = _imagenetv1()
    model.train().to(device)

    #params_to_update = model.freeze()

    criterion = torch.nn.BCELoss()
    optimizer = _optimizer(model, conf, criterion)

    for epoch in range(conf.train.epochs):
        print('Epoch {}/{}'.format(epoch + 1, conf.train.epochs))
        print('-' * 10)

        for phase in ['train', 'val']:

            running_loss = 0.0
            running_recall = 0
            running_prec = 0

            data_loader = data_loaders[phase]

            # Iterate over data.
            n = len(data_loader.dataset) // data_loader.batch_size
            count = 0
            for inputs, labels in data_loader:#tqdm(data_loader, total=n):

                # Data from generator
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Make gradients zero
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    # Get loss & prediction
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    preds = torch.where(outputs > 0.5, 1, 0)

                    recall = torch.where(labels > 0.5, preds, torch.zeros_like(preds))
                    recall = torch.sum(recall)/torch.sum(labels)

                    prec = torch.where(preds > 0.5, labels, torch.zeros_like(labels))
                    prec = torch.sum(prec)/torch.sum(preds)

                    # Backpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Update loss & acc for epoch end
                running_loss += loss.item() 
                running_recall += recall 
                running_prec += prec 

                # Quick feedback on loss
                count += 1
                l = running_loss / count 
                recall = running_recall / count
                prec = running_prec / count
                print('Loss: {:2.2f} Recall: {:2.2f} Precision: {:2.2f}'.format(l, recall, prec))


            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(
                data_loaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def run(conf, _finetune):

    # Create generators
    datasets = {
        x: CelebAGenerator.from_conf(conf, is_train=x == 'train')
        for x in ['train', 'val']
    }
    data_loaders = {
        x: DataLoader(datasets[x],
                      batch_size=conf.train.batch_size,
                      shuffle=x == 'train',
                      num_workers=6,
                      prefetch_factor=10,
                      pin_memory=True)
        for x in ['train', 'val']
    }

    # Get new training directory and save weights before training
    save_folder = create_train_directory(conf.path.models_root)

    if _finetune:
        finetune(data_loaders, save_folder, conf)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)

    conf = get_config(args.config)
    run(conf, args.finetune)


if __name__ == '__main__':
    main()