import os
import sys
import argparse
import tensorflow as tf
from datetime import datetime

if './' not in sys.path:
    sys.path.append('./')

from src.lucid.visualizer import Visualizer

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pb', help='The path to the pb model file')
    parser.add_argument('--save-dir', '-d', help='The path to the storage folder', default='results')
    parser.add_argument('--layers', '-l', help='Path to the txt file containing the layers\'s names', default='archive/relu_layer_info.txt')
    parser.add_argument('--no-date', help='Do not add extra date folder', action='store_false', dest='use_date')
    return parser.parse_args(args)

def _create_dir(save_dir, subfolder=None, use_date=True):
    os.makedirs(save_dir, exist_ok=True)
    if subfolder is not None:
        if use_date:
            date = datetime.now().strftime('%Y-%m-%d-%H%M%S')
            subfolder = subfolder+'-'+date
        save_dir = os.path.join(save_dir, subfolder)
        os.mkdir(save_dir)
    return save_dir

def _get_layer_info(path):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        layer_name, n_channels = line.split(' ')
        n_channels = int(n_channels)
        data.append([layer_name, n_channels])
    return data


def run(pb_path, save_dir, layer_info_path, use_date):
    # Create save directory
    filename = os.path.split(pb_path)[1][:-3]
    save_dir = _create_dir(save_dir, subfolder=filename, use_date=use_date)

    # Retrieve layers' names and number of channels
    layer_info =  _get_layer_info(layer_info_path)

    # Run visualization
    visualizer = Visualizer(pb_path)
    visualizer.evaluate_on_layer_info(layer_info, save_dir)




def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.pb, args.save_dir, args.layers, args.use_date)

if __name__ == '__main__':
    main()