import os
import sys
from tqdm import tqdm
import argparse
import cv2
from datetime import datetime
from celebalucid import load_model

if './' not in sys.path:
    sys.path.append('./')

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('model_name', help='Name of the model')
    parser.add_argument('--save-dir', '-d', help='The path to the storage folder', default='results')
    return parser.parse_args(args)

def _create_dir(save_dir, subfolder=None):
    os.makedirs(save_dir, exist_ok=True)
    if subfolder is not None:
        date = datetime.now().strftime('%Y-%m-%d-%H%M%S')
        subfolder = subfolder+'-'+date
        save_dir = os.path.join(save_dir, subfolder)
        os.mkdir(save_dir)
    return save_dir

def run(model_name, save_dir):

    # Create save directory
    save_dir = _create_dir(save_dir, subfolder=model_name)

    model = load_model(model_name)
    reverse_layer_info = model.layer_info[::-1]
    for layer, n_channels in tqdm(reverse_layer_info):
        for i in tqdm(range(n_channels)):
            layer_n_channel = layer+':'+str(i)
            img = model.lucid(layer_n_channel)

            local_dir = os.path.join(save_dir, layer)
            os.makedirs(local_dir, exist_ok=True)
            path = os.path.join(local_dir, str(i)+'.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, img)




def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.model_name, args.save_dir)

if __name__ == '__main__':
    main()