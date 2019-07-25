import numpy as np
from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence

class ImageSequence(Sequence):
    def __init__(self, x_set, y_set, input_shape, batch_size, normalize=False):
        self.x, self.y = x_set, y_set
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.normalize = normalize

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        def load_image(file_name, normalize=False):
            img = resize(imread(file_name), self.input_shape)
            if normalize:
                img = img / 255.0
            return img
        return np.array([load_image(file_name, normalize=self.normalize) for file_name in batch_x]), np.array(batch_y)


class AugmenterSequence(Sequence):
    def __init__(self, sequence, augmenter):
        self.sequence = sequence
        self.augmenter = augmenter
    
    def __len__(self):
        return self.sequence.__len__()

    def __getitem__(self, idx):
        batch_x, batch_y = self.sequence.__getitem__(idx)
        return self.augmenter(batch_x), batch_y
