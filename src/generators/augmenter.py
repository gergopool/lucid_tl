from imgaug import augmenters as iaa

# TODO: Make this class flexible for config
class CelebAAugmenter:

    def __init__(self):
        often = lambda aug: iaa.Sometimes(0.8, aug)
        self.seq = iaa.Sequential([
            often(
                iaa.Affine(
                    scale=(0.9, 1.2),
                    translate_percent=(-0.2, 0.2),
                    rotate=(-10, 10), 
                    shear=(-10, 10),
                    mode="edge"
                    ),
            ),
            iaa.Fliplr(0.5)
        ])

    def __call__(self, X):
        return self.seq(images=X)