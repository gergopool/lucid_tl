import keras
from keras import layers
from src.models.inception_v1 import InceptionV1


class CelebAModel:
    def __init__(self,
                 n_features=40,
                 input_shape=(224, 224, 3),
                 include_top=False,
                 weights='imagenet',
                 model_path=None,
                 lr=1e-4,
                 optimizer='adam',
                 loss='binary_crossentropy',
                 **kwargs):
        self.model = self._build(n_features, input_shape, include_top, weights,
                                 model_path, **kwargs)
        self.lr = lr
        self.optimizer = optimizer
        self.loss = loss

    @classmethod
    def from_conf(cls, conf):
        return cls(n_features=conf.train.n_features,
                   input_shape=conf.train.image_shape,
                   include_top=False,
                   weights='imagenet',
                   model_path=conf.path.finetuned,
                   lr=conf.train.lr,
                   optimizer=conf.train.optimizer,
                   loss=conf.train.loss)

    def compile(self, optimizer=None):
        optimizer = self.optimizer if optimizer is None else optimizer
        if optimizer.lower() == 'sgd':
            opt = keras.optimizers.SGD(lr=self.lr, momentum=0.9)
        elif optimizer.lower() == 'adam':
            opt = keras.optimizers.Adam(lr=self.lr)
        self.model.compile(opt,
                           loss=self.loss,
                           metrics=['accuracy'])

    def freeze(self):
        ''' Freeze the CNN layers, everything but the last random layers '''
        trainable = False
        for layer in self.model.layers:
            # Freeze until this layer
            if layer.name == 'bottleneck':
                trainable = True
            layer.trainable = trainable

    def unfreeze(self):
        ''' Unfreeze the entire network '''
        for layer in self.model.layers:
            layer.trainable = True

    def _build(self, n_features, input_shape, include_top, weights, model_path,
               **kwargs):
        ''' Create keras model '''

        # Load if h5 path is given, else create it
        if model_path is not None:
            print('*** Loading in model ***')
            return keras.models.load_model(model_path)
        else:
            print('*** Creating new model ***')
            base_model = InceptionV1(input_shape=input_shape,
                                     include_top=include_top,
                                     weights=weights,
                                     **kwargs)

            inputs = base_model.inputs
            x = base_model.outputs[0]

            # Inception V1 ending
            x = layers.AveragePooling2D((7, 7),
                                        strides=(1, 1),
                                        padding='valid',
                                        name='bottleneck')(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Conv2D(n_features, (1, 1),
                              strides=(1, 1),
                              padding='valid',
                              use_bias=True,
                              name='Logits')(x)
            x = layers.Flatten(name='Logits_flat')(x)
            x = layers.Activation('sigmoid', name='Predictions')(x)

            return keras.models.Model(inputs=inputs, outputs=[x])
