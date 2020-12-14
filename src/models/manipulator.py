import os
import numpy as np
from dotmap import DotMap
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from lucent.optvis import transform, param, render

from src.models.inceptionv1 import InceptionV1
from src.utils.io import load_layer_info


class ModelManipulator(InceptionV1):
    def __init__(self,
                 pt_url):
        super(ModelManipulator, self).__init__(n_features=40,
                                               pretrained=False,
                                               redirected_ReLU=True)
        self._load_weights_from_url(pt_url)
        self.layer_info = load_layer_info()
        self.weights = self._load_weights()
        self.neurons = {}
        self._activations = {}
        self._register_activation_fw_hooks()

        # Assign device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device).eval()

    def lucid(self, layer_n_channel, size=224, thresholds=[512], progress=False):
        layer_n_channel = self._correct_layer_n_channel(layer_n_channel)
        transforms = transform.standard_transforms.copy()
        transforms.append(lambda x: x * 255 - 117)
        def param_f(): return param.image(224, fft=True, decorrelate=True)
        try:
            img = render.render_vis(self,
                                    layer_n_channel,
                                    show_image=False,
                                    preprocess=False,
                                    progress=progress,
                                    transforms=transforms,
                                    thresholds=thresholds,
                                    param_f=param_f)[0][0]
        except AssertionError:
            raise AssertionError(
                'Invalid layer {}. Retrieve the list of layers with `model.layer_info`.'.format(
                    layer_n_channel)
            )

        img = (img*255).astype(np.uint8)
        return img

    def stream(self, x):
        x = x.to(self.device)
        self.forward(x)
        self.neurons = DotMap(self._activations)

    def set_weights(self, layer, reference, mode='both'):
        valid_modes = ['both', 'weight', 'bias']
        if mode not in valid_modes:
            raise ValueError('Invalid mode: {}. Please provide a mode from {}'\
                             .format(mode, ', '.join(valid_modes)))
        if mode == 'both':
            self.set_weights(layer, reference, 'weight')
            self.set_weights(layer, reference, 'bias')
            return
        layer = self._correct_layer_n_channel(layer)
        layer, neuron_i = self._split_layer_name(layer)
        targets = self._extract_target_weights(reference, layer, neuron_i, mode)
        state_dict = self.state_dict()
        new_state_dict = self._change_state_dict(state_dict, layer,
                                                 neuron_i, targets, mode)
        self.load_state_dict(new_state_dict)


    # ========================================================================
    # Private functions
    # ========================================================================

    def _extract_target_weights(self, reference, layer, i, mode='weight'):
        if type(reference) == type(self):
            target = self._get_layer(reference, layer, i, mode)
        else:
            baseline = self._get_layer(self, layer, i, mode)
            target = torch.ones_like(baseline, dtype=baseline.dtype) * reference
        return target
            

    def _change_state_dict(self, state_dict, layer, i, value, mode='weight'):
        if i is None:
            state_dict[layer+'.'+mode] = value
        else:
            state_dict[layer+'.'+mode][i] = value
        return state_dict

    def _get_layer(self, model, layer, i, mode='weight'):
        states = model.state_dict()
        layer = states[layer+'.'+mode]
        if i is not None:
            layer = layer[i]
        return layer

    def _split_layer_name(self, layer):
        if ':' not in layer:
            return layer, None
        else:
            layer, neuron_i = layer.split(':')
            neuron_i = int(neuron_i)
            return layer, neuron_i

    def _correct_layer_n_channel(self, layer_n_channel):
        if ':' in layer_n_channel:
            layer, channel = layer_n_channel.split(':')
            layer += '_pre_relu_conv' if layer != 'logits' else ''
            return ':'.join([layer, channel])
        else:
            layer = layer_n_channel
            layer += '_pre_relu_conv' if layer != 'logits' else ''
            return layer

    def _register_activation_fw_hooks(self):
        for name, module in self.named_modules():
            if name.endswith('_pre_relu_conv') or name == 'logits':
                name = name.replace('_pre_relu_conv', '')
                args = partial(self._save_activation, name)
                module.register_forward_hook(args)

    def _save_activation(self, name, module, m_in, m_out):
        self._activations[name] = m_out

    def _load_weights(self):
        extracted_weights = DotMap()
        for name, weights in self.named_parameters():
            layer_name, weight_format = name.split('.')
            weight_format = weight_format[0]
            layer_name = layer_name.replace('_pre_relu_conv', '')
            extracted_weights[layer_name][weight_format] = weights.detach(
            ).cpu().numpy()

        return extracted_weights

    def _load_weights_from_url(self, url):
        self.load_state_dict(torch.hub.load_state_dict_from_url(
            url, progress=True), strict=False)
