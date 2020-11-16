import os
import sys
import argparse
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from tensorflow.python import framework as fw
import keras

def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('model', help='The path to the h5 model file')
    return parser.parse_args(args)

def save_pb(model, path):
    # Retrieve root folder and filename
    folder, filename = os.path.split(path)

    keras.backend.set_learning_phase(0)
    with keras.backend.get_session() as sess:

        # Change output names to access them easies
        output_node_names = []
        for i in range(len(model.outputs)):
            future_name = 'output_'+str(i)
            output_node_names.append(future_name)
            _ = tf.identity(model.outputs[i], name=future_name)
        print('Output nodes names are: ', output_node_names)

        # Save graph to pb
        constant_graph = fw.graph_util.convert_variables_to_constants(
            sess, sess.graph.as_graph_def(), output_node_names)
        fw.graph_io.write_graph(constant_graph,
                                folder,
                                filename,
                                as_text=False)

    # Informing user
    print('Saved the constant graph (ready for inference) at: ', path)

    keras.backend.set_learning_phase(1)


def run(h5_model_path):
    # Loading in the keras h5 model, result of training
    keras_model = keras.models.load_model(h5_model_path)
    # Determining pb path. This covers names which contains dot in the middle
    # and every h5 format (hdf5 and h5)
    pb_model_path = '.'.join(h5_model_path.split('.')[:-1])+'.pb'

    save_pb(keras_model, pb_model_path)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.model)


if __name__ == '__main__':
    main()