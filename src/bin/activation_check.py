import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse
import cv2
import gc

if './' not in sys.path:
    sys.path.append('./')

from src.utils.maths import cka, gram_linear
from src.eval import CelebAController
from src.generators import CelebAGenerator
from src.utils import get_config

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('n', help='Number of images', type=int)
    parser.add_argument('--pre-pb', help='The path to the pre train pb model file', default='archive/imagenet.pb')
    parser.add_argument('--post-pb', help='The path to the post train pb model file', default='archive/sgd.pb')
    parser.add_argument('--batch-size', help='Batch size', default=25)
    parser.add_argument('--config', help='Path to config file', default='config/train/test_only.ini')
    parser.add_argument('--layer-info', help='Path to layer info', default='archive/relu_layer_info.txt')
    parser.add_argument('--save-dir', help='Path to save directory', default='results/activation/')
    return parser.parse_args(args)

def _get_layer_info(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        layer_name, n_channels = line.split(' ')
        n_channels = int(n_channels)
        data.append([layer_name, n_channels])
    return data

def _get_gen(conf, batch_size):
    gen = CelebAGenerator.from_conf(conf, is_train=False)
    gen.batch_size = batch_size
    gen.shuffle = True
    return gen

def _get_models(conf, models, batch_size):
    shape = conf.train.image_shape
    models = {name:CelebAController(path, shape, batch_size=batch_size)
            for (name,path)
            in models.items()}
    return models

def simulate(n, all_imgs, models, layers):

    results = []

    for layer in layers:
        preds = {model_name:[] for model_name in models}
        for model_name, model in models.items():
            # Init
            model.set_outputs([layer])

            # Predict
            for i in range(n):
                imgs= all_imgs[i]
                pred = model.predict(imgs, preprocess=False)[0]
                pred = pred.reshape(len(imgs), -1)
                preds[model_name].extend(pred)
                
            # Get matrix
            preds[model_name] = np.array(preds[model_name])
        
        X = preds['pre']
        Y = preds['post']
        dist = cka(gram_linear(X), gram_linear(Y))
        results.append(dist)
        
        preds = None
        X = None
        Y = None
        dist = None

        gc.collect()

    series = pd.Series(results, index=layers)
    df = pd.DataFrame(series).transpose()
    return df

def _provide_save_path(save_dir, n):
    # Create new folder with current date
    date = datetime.now().strftime('%Y-%m-%d')
    folder = os.path.join(save_dir, date)
    filepath = os.path.join(folder, str(n)+'.csv')
    os.makedirs(folder, exist_ok=True)
    return filepath

def _get_imgs(gen, n_batches):
    imgs = []
    for i in range(n_batches):
        imgs.append(gen[i][0])
    return imgs

def run(args):

    models = {'pre' : args.pre_pb, 'post' : args.post_pb}
    conf = get_config(args.config)
    layers = [x[0] for x in _get_layer_info(args.layer_info)]

    n_batches = args.n // args.batch_size
    gen = _get_gen(conf, args.batch_size)
    imgs = _get_imgs(gen, n_batches)
    models = _get_models(conf, models, args.batch_size)
    save_path = _provide_save_path(args.save_dir, args.n)

    df = pd.DataFrame()
    for i in tqdm(range(100)):
        gen.on_epoch_end() # shuffle
        imgs = _get_imgs(gen, n_batches)
        results = simulate(n_batches, imgs, models, layers)
        df = df.append(results)
        df.to_csv(save_path, index=False)

        


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args)

if __name__ == '__main__':
    main()