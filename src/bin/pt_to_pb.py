import os
import sys
import argparse
import torch
import onnx
from torch.autograd import Variable

if './' not in sys.path:
    sys.path.append('./')

from src.models.celeb_a_torch import CelebAModel

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('model', help='The path to the p5 model file')
    parser.add_argument('--input-size', '-s', type=int, default=224)
    return parser.parse_args(args)

def run(pt_model_path, input_size):
    
    # Paths
    workdir = os.path.split(pt_model_path)
    onnx_path = pt_model_path.replace('.pth', '.onnx').replace('.pt', '.onnx')
    pb_path = onnx_path.replace('.onnx', '.pb')

    # Load pytorch
    print('Loading in pytorch.')
    model = CelebAModel(pretrained=False)
    model.load_state_dict(torch.load(pt_model_path))
    model.eval()

    # Save onnx
    dummy_input = Variable(torch.randn(1, 3, input_size, input_size))
    print('Saving onnx.')
    torch.onnx.export(model, dummy_input, onnx_path)

    # Load & convert onnx, then save to pb
    print('Loading in onnx.')
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    print('Saving pb.')
    tf_rep.export_graph(pb_path)

    # Clear up
    print('Removing onnx.')
    os.remove(onnx_path)

    print('Done.')

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.model, args.input_size)


if __name__ == '__main__':
    main()