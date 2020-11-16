import tensorflow as tf
from tqdm import trange
import numpy as np
import pandas as pd
from abc import ABC
from threading import Lock
from tensorflow.python.framework import tensor_util


# Abstract Base Class
class GeneralPbController(ABC):
    '''
        This class is an abstract class that serves the functionality of
         - Reading and preprocessing images
         - Executing the neural network on desired outputs
         - Decoding the output of the neural network.

        Moreover it is responsible for executing every PbPredictor
        insances while providing a thread-safe solution for GPU.
    '''

    gpu_lock = Lock()

    def __init__(self, pb_path, x, target_shape, batch_size=32):
        ''' Default initialization '''
        self.nn_evaluator = PbExecutor(pb_path, x)
        self.target_shape = target_shape
        self.batch_size = batch_size
        self.index_map = None

    @classmethod
    def from_conf(cls, conf):
        ''' Initialization from config file '''
        keys = conf.__dict__
        batch_size = 32 if 'batch_size' not in keys else conf.batch_size
        return cls(conf.x, conf.y, conf.target_shape, batch_size)

    @classmethod
    def from_dict(cls, conf):
        ''' Initialization from dictionary '''
        batch_size = 32 if 'batch_size' not in conf else conf['batch_size']
        return cls(conf['x'], conf['y'], conf['target_shape'], batch_size)

    # ==============================================================
    # PUBLIC
    # ==============================================================

    def set_outputs(self, y=None, y_rule=None):
        self.nn_evaluator.set_outputs(y, y_rule)
        self.index_map = self._create_index_map()

    def predict(self, imgs, verbose=0, preprocess=True):
        '''Run a full prediction: preprocess, neural net, decode'''
        results = []
        range_func = trange if verbose else range
        for start in range_func(0, len(imgs), self.batch_size):
            end = min(len(imgs), start + self.batch_size)
            batch_imgs = imgs[start:end]
            nn_imgs = self.preprocess(batch_imgs) if preprocess else batch_imgs
            nn_results = self._evaluate_nn(nn_imgs)
            predictions = self.decode(nn_results)
            results.extend(predictions)
        return results

    def preprocess(self, img_paths):
        ''' Preprocessing images to serve the right input format '''
        raise NotImplementedError('Preprocess not implemented.')

    def decode(self, predictions):
        raise NotImplementedError('Decoding not implemented.')

    def filter_activations(self, batch_neurons, global_i=None, **kwargs):
        layer_i, channel_i = self._get_layer_n_channel_index(
            global_i, **kwargs)
        activations = batch_neurons[layer_i][..., channel_i]
        return activations

    def get_weights(self, global_i=None, **kwargs):
        layer_i, channel_i = self._get_layer_n_channel_index(
            global_i, **kwargs)
        return self.nn_evaluator.weights[layer_i][..., channel_i]

    def get_layer_info(self):
        y_names = self.nn_evaluator.y_names
        channel_counts = self.nn_evaluator.channel_counts
        info = {name: c for (name, c) in zip(y_names, channel_counts)}
        return info

    def get_no_neurons(self):
        return np.sum(self.nn_evaluator.channel_counts)

    # ==============================================================
    # PRIVATE
    # ==============================================================

    def _get_layer_n_channel_index(self,
                                   global_i=None,
                                   layer_i=None,
                                   layer_name=None,
                                   channel_i=None):

        if global_i is not None:
            condition = self.index_map.global_i == global_i
            match_df = self.index_map[condition]
            if not len(match_df):
                raise ValueError(
                    'Global index {} does not exist in network.'.format(
                        global_i))
            match = match_df.iloc[0]
            layer_i = match.layer_i
            channel_i = match.channel_i

        if layer_name is not None:
            condition = self.index_map.layer_name == layer_name
            match_df = self.index_map[condition]
            if not len(match_df):
                raise ValueError(
                    'Layer name {} does not exist in network.'.format(
                        global_i))
            match = match_df.iloc[0]
            layer_i = match.layer_i

        if channel_i is None:
            raise ValueError('Please also provide the relative index \
                    of the neuron within the layer.')

        return layer_i, channel_i

    def _create_index_map(self):
        y_names = self.nn_evaluator.y_names
        channel_counts = self.nn_evaluator.channel_counts
        global_i = 0
        dataframes = []
        for layer_i, (y_name,
                      channel_count) in enumerate(zip(y_names,
                                                      channel_counts)):
            counts = np.arange(channel_count)
            df = pd.DataFrame({
                'layer_name': y_name,
                'layer_i': layer_i,
                'global_i': counts + global_i,
                'channel_i': counts
            })
            dataframes.append(df)
            global_i += channel_count
        merged = pd.concat(dataframes, axis=0)
        return merged

    def _evaluate_nn(self, images):
        with GeneralPbController.gpu_lock:
            encoded_results = self.nn_evaluator.predict(images)
        return encoded_results

    def close(self):
        self.nn_evaluator.close()


class PbExecutor:
    """This class executes a frozen graph and predicts on certain inputs.
    
    Attributes:
        pb_path (string): Path to the pb file.
        x (Tensor): The input tensor
        y (Tensor): The output tensor(s)
    """
    def __init__(self, pb_path, x):
        self.graph, self.graph_def = self._load_graph(pb_path)
        self.x = self._load_tensor(x)
        self.y = None
        self.y_names = []
        self.weights = []
        self.sess = self._get_session()

    def set_outputs(self, y_names=None, y_rule=None):
        if y_names is not None:
            self.y = self._load_tensor(y_names)
        elif y_rule is not None:
            y_names = [
                op.name for op in self.graph.get_operations()
                if y_rule(op.name)
            ]
            self.y = self._load_tensor(y_names)

        self.y_names = y_names
        self.channel_counts = self._get_channel_count()

        # Get weights
        operations = [
            op for op in self.graph.get_operations()
            if op.name.endswith('kernel')
        ]
        self.weights = self.sess.run([op.outputs[0] for op in operations])

    def _get_channel_count(self):
        results = []
        for layer in self.y:
            results.append(int(layer.shape[-1]))
        return results

    def predict(self, inputs):
        """Execute prediction on input
        
        Args:
            inputs: The inputs or single input
        
        Returns:
            Numpy.NDarray: The results
        """
        if self.y is None:
            raise ValueError('First define the network\'s outputs!')

        if isinstance(self.x, list):
            if not isinstance(inputs, list) or len(inputs) != len(self.x):
                raise ValueError('Number of inputs must match with config.')
            feed_dict = {self.x[i]: inputs[i] for i in range(len(self.x))}
        else:
            feed_dict = {self.x: inputs}

        y_pred = self.sess.run(self.y, feed_dict=feed_dict)
        return y_pred

    def close(self):
        self.sess.close()

    def _load_tensor(self, nodename):
        get_tensor = self.graph.get_tensor_by_name
        if isinstance(nodename, list):
            tensor = [get_tensor(n + ':0') for n in nodename]
        else:
            tensor = get_tensor(nodename + ':0')
        return tensor

    def _load_graph(self, frozen_graph_filename):
        """Summary
        
        Args:
            frozen_graph_filename (string): Path to the pb file
        
        Returns:
            Tensorflow.Graph: Tensorflow graph definition
        """
        with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        return graph, graph_def

    def _get_session(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        return tf.compat.v1.Session(graph=self.graph, config=config)
