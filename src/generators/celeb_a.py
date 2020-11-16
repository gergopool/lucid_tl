import numpy as np
import cv2
import pandas as pd
import os
import keras

from src.generators.augmenter import CelebAAugmenter
from src.utils.vis import first_look_on_imgs
from src.utils import get_preprocess_func


class CelebAGenerator(keras.utils.Sequence):
    """ Generator for lucid InceptionV1 GoogLeNet
        Args:
            csv (str):
                The path to the csv, placed in the same folder
                which holds the images.
            target_shape (tuple, optional):
                The input shape of the neural network.
                Defaults to (224, 224, 3).
            batch_size (int, optional):
                (Maximum) batch size. Defaults to 32.
            shuffle (bool, optional):
                Either to shuffle the data after each epoch and at the
                beginning of the training. Defaults to False.
            augment_func (func, optional):
                Augmentation function. Defaults to None.
            preprocess_func (str, optional):
                Image preprocessing function, applied on RGB uint8 input.
                Can be either 'imagenet' or 'robi'.
                Defaults to 'imagenet'.
    """
    def __init__(self,
                 csv,
                 target_shape=(224, 224, 3),
                 batch_size=32,
                 shuffle=False,
                 augment_func=None,
                 preprocess_func='imagenet'):

        # Read in csv and define root directory
        self.df = pd.read_csv(csv)
        self.root = os.path.split(csv)[0]

        # Convert relative paths to absolute
        self.df.img = self.df.img.apply(lambda x: os.path.join(self.root, x))

        self.target_shape = target_shape
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment_func = augment_func
        self.preprocess_func = get_preprocess_func(preprocess_func)

    @classmethod
    def from_conf(cls, conf, is_train):
        shuffle = is_train
        augment_func = CelebAAugmenter() if is_train else None
        csv = conf.path.train_df if is_train else conf.path.test_df
        return cls(csv=csv,
                   target_shape=conf.train.image_shape,
                   batch_size=conf.train.batch_size,
                   shuffle=shuffle,
                   augment_func=augment_func,
                   preprocess_func=conf.train.preprocess_func)

    def __len__(self):
        ''' The length of the generator. Df size / batch_size '''
        return int(np.ceil(len(self.df) / self.batch_size))

    def __getitem__(self, i):
        ''' Get i^th item of the generator as gen[i] '''
        # Get i^th batch of data
        batch_data = self._get_slice(i)

        # Input
        batch_paths = batch_data.img
        X = [cv2.imread(path) for path in batch_paths]
        if self.augment_func is not None:
            X = self.augment_func(X)

        X = self._preprocess(X)

        # Output
        Y = batch_data.drop('img', axis=1).values
        Y = (Y + 1) / 2.  # [-1,1] data to [0,1]

        return X, Y

    def visualize(self, i, rows=4, cols=8):
        ''' Quick look on batch of data by plotting them '''
        assert rows * cols == self.batch_size, \
               'Please ensure rows * cols = batch_size'
        X, _ = self[i]
        X = (X + 1) / 2  # NOTE: This depends on the preprocessing function
        first_look_on_imgs(X, rows, cols)

    def on_epoch_end(self):
        ''' Run this function whenever an epoch ends. Shuffle! '''
        if self.shuffle:
            self.df = self.df.sample(frac=1)

    # ================================================================
    # Helper functions
    # ================================================================

    def _get_slice(self, i):
        ''' Crop dataframe to indexed batch '''
        start = i * self.batch_size
        end = min((i + 1) * self.batch_size, len(self.df))
        df = self.df.iloc[start:end]
        return df

    def _preprocess(self, imgs):
        ''' Convert images to default RGB, resized format and apply
            preprocessing function.
        '''
        rgb_imgs = []
        # Define target height and width
        h, w = self.target_shape[:2]
        # Read in images as RGB
        for img in imgs:
            img = cv2.resize(img, (w, h))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_imgs.append(img)
        rgb_imgs = np.array(rgb_imgs)

        # Apply preprocessing function
        preprocessed_imgs = self.preprocess_func(rgb_imgs)

        return preprocessed_imgs
