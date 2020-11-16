import cv2
import sys
import numpy as np
import time
import os
if './' not in sys.path:
    sys.path.append('./')
from src.eval.general_pb_controller import GeneralPbController


class CelebAController(GeneralPbController):
    def __init__(self, pb_path, target_shape, x='input_1', **kwargs):
        super().__init__(pb_path, x, target_shape=target_shape, **kwargs)
        self._set_default_outputs()

    def _set_default_outputs(self):
        default_func = lambda x: x.endswith('convolution')
        self.set_outputs(y_rule=default_func)

    def preprocess(self, imgs):
        # If image paths are given
        if isinstance(imgs[0], str) and os.path.isfile(imgs[0]):
            imgs = [cv2.imread(img) for img in imgs]
            imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]

        # Resize and normalize
        h, w = self.target_shape[:2]
        imgs = [cv2.resize(img, (w, h)) for img in imgs]
        imgs = np.array(imgs) / 255.
        return imgs

    def decode(self, predictions):
        return predictions



if __name__ == "__main__":
    import glob
    paths = glob.glob('res/images/img_align_celeba/*.jpg')[:32]
    predictor = CelebAController('res/models/200.pb', (299, 299, 3))

    preds = predictor.predict(paths)
    print(len(preds))
    #print(predictor.filter_neurons(preds, global_i=37))
    print(predictor.get_weights(1200).shape)
    #print(preds[-3])
