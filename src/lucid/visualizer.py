from tqdm import tqdm
import os
import cv2
import numpy as np
from lucid.modelzoo.vision_base import Model
import lucid.optvis.param as param
import lucid.optvis.render as render
import lucid.optvis.objectives as objectives


class Visualizer:
    def __init__(self,
                 pb_path,
                 image_shape=[224, 224, 3],
                 image_value_range=(-1, 1),
                 input_name='input_1',
                 threshold=2048,
                 scale=224):
        self.model = self._create_model(pb_path, image_shape,
                                        image_value_range, input_name)
        self.threshold = threshold
        self.param_f = lambda: param.image(scale, fft=True, decorrelate=True)

    def _create_model(self, pb_path, image_shape, image_value_range,
                      input_name):
        model = Model()
        model.model_path = pb_path
        model.image_shape = image_shape
        model.image_value_range = image_value_range
        model.input_name = input_name
        model.load_graphdef()
        return model

    def _get_objective(self, layer, index=0):
        return objectives.channel(layer, index)

    def evaluate_on_layer_info(self, layer_info, save_root):
        for layer_name, n_channels in tqdm(layer_info):

            # Create subfolder
            subfolder_name = layer_name.replace('/','--')
            save_dir = os.path.join(save_root, subfolder_name)
            os.makedirs(save_dir, exist_ok=True)

            for i in range(n_channels):
                filepath = os.path.join(save_dir, str(i)+'.jpg')
                image = self(layer_name, i)
                image = np.round(image * 255).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, image)


    def __call__(self, layer_name, channel_index):
        obj = objectives.channel(layer_name, channel_index)
        image = render.render_vis(self.model,
                                  obj,
                                  param_f=self.param_f,
                                  thresholds=[self.threshold],
                                  verbose=False)
        return np.array(image[0][0])