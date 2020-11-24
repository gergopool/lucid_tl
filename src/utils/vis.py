from matplotlib import pyplot as plt
import cv2
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import multiprocessing


def first_look_on_imgs(imgs, rows=4, cols=8):
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    for i, img in enumerate(imgs):
        ax = fig.add_subplot(rows, cols, i + 1, xticks=[], yticks=[])
        ax.imshow(img)
    plt.show()

class VisController:

    LAYER_INFO_PATH = 'archive/relu_layer_info.txt'
    ROOT = 'res/lucid_vis_img'
    NO_IMG_PATH = 'res/images/no_img.jpg'
    MODELS = ['imagenet', 'sgd', 'adam', 'adam_overfit']
    VIS_TYPES = ['lucid', 'stylegan']
    IMG_SIZE = (224, 224, 3)
    N_CORES = min(multiprocessing.cpu_count(), 4)

    def __init__(self, **kwargs):
        self.root = kwargs.get('root', VisController.ROOT)
        self.img_size = kwargs.get('img_size', VisController.IMG_SIZE)
        no_img_path = kwargs.get('no_img_path', VisController.NO_IMG_PATH)
        layer_info_path = kwargs.get('layer_info_path',
                                     VisController.LAYER_INFO_PATH)
        self.models = kwargs.get('models', VisController.MODELS)
        self.vis_types = kwargs.get('vis_types', VisController.VIS_TYPES)
        self.n_cores = kwargs.get('n_cores', VisController.N_CORES)

        self.layer_info = self._get_layer_info(layer_info_path)
        self.df = self._generate_df(self.layer_info)
        self.no_img = self._read_in_no_img(no_img_path)

        tqdm.pandas()
        # pandarallel.initialize(nb_workers=self.n_cores, progress_bar=True)

    def _generate_df(self, layer_info):
        data = dict(model=[], layer=[], channel=[], vis_type=[])
        for model in self.models:
            for layer, n_channels in layer_info:
                for channel in range(n_channels):
                    for vis_type in self.vis_types:
                        data['model'].append(model)
                        data['layer'].append(layer)
                        data['channel'].append(channel)
                        data['vis_type'].append(vis_type)

        df = pd.DataFrame(data)
        df['img'] = df.apply(lambda row: os.path.join(
            self.root, row.vis_type, row.model, row.layer.replace('/', '--'),
            str(row.channel) + '.jpg'),
                             axis=1)
        return df

    def _isnull(self, v):
        b = False
        b = b or v is None
        b = b or v == 'None'
        b = b or v == ''
        b = b or v == -1
        returnb

    def get_images(self, force=False, verbose=False, **kwargs):
        w = True
        for column, value in kwargs.items():
            if self._isnull(value):
                continue
            try:
                look_up_series = self.df[column].astype(str).str.lower()
                look_up_value = str(value).lower()
                new_w = look_up_series == look_up_value
                w = w & new_w
            except KeyError:
                raise KeyError('No filter exists for {}'.format(column))

        filtered = self.df[w]
        if len(filtered) == 0:
            raise ValueError('No match was found!')
        if len(filtered) > 500 and not force:
            raise ValueError(
                'You want to acces {} pictures. If you are sure you want to continue, pass force=True as argument.'
                .format(len(filtered)))

        apply = filtered.progress_apply if verbose else filtered.apply
        data = apply(lambda x: self.load_image(x.img), axis=1)
        info = filtered.reset_index(drop=True)
        info['success'] = [d[0] for d in data]
        images = [d[1] for d in data]
        images = np.array(images, dtype='uint8')

        return info, images

    def load_image(self, path):
        img = cv2.imread(path)
        found = img is not None
        img = self._to_rgb(img) if found else self.no_img
        img = self._resize(img)
        return found, img

    def _to_rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _resize(self, img):
        h, w = self.img_size[:2]
        img = cv2.resize(img, (w, h))
        return img

    def _get_layer_info(self, path):
        data = []
        with open(path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            layer, n_channels = line.split(' ')
            n_channels = int(n_channels)
            data.append([layer, n_channels])
        return data

    def _read_in_no_img(self, path):
        img = cv2.imread(path)
        img = self._to_rgb(img)
        return img

vis = VisController()