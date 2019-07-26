import os
import json
import pickle
import shutil
import numpy as np

from skimage.io import imsave
from keras.utils import to_categorical

from ..dataset import Dataset
from ..generators import ImageSequence

class CIFAR10(Dataset):
    nb_classes = 10
    class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    input_shape = (32, 32, 3)

    @staticmethod
    def download(download_path):
        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'download.sh')
        os.system('sh %s %s' % (script_path, download_path))

    @staticmethod
    def init(data_path, download_path):
        def unpickle(file):
            with open(file, 'rb') as fo:
                return pickle.load(fo, encoding='bytes')

        os.makedirs(data_path)

        train_batches = [ 'data_batch_' + str(i) for i in range(1,6) ]
        for i in range(5):
            dic = unpickle( os.path.join(download_path, 'cifar-10-batches-py/' + train_batches[i]) )
            for j in range(10000):
                data = np.resize(dic[b'data'][j], CIFAR10.input_shape[-1:] + CIFAR10.input_shape[:-1])
                data = np.moveaxis(data, source=0, destination=-1)
                imsave(os.path.join(data_path, '%db_%di_%dc.jpg' % (i, j, dic[b'labels'][j])), data)

        test_batch = 'test_batch'
        dic = unpickle( os.path.join(download_path, 'cifar-10-batches-py/' + test_batch) )
        for j in range(10000):
            data = np.resize(dic[b'data'][j], CIFAR10.input_shape[-1:] + CIFAR10.input_shape[:-1])
            data = np.moveaxis(data, source=0, destination=-1)
            imsave(os.path.join(data_path, '%db_%di_%dc.jpg' % (6, j, dic[b'labels'][j])), data)

    @staticmethod
    def split(data_path, conf, split=None):
        sets = conf['sets']
        files = []
        for f in os.listdir(data_path):
            files.append(f)

        if split is None:
            ratios = conf['ratios']
            samples = np.asarray(ratios) * len(files)
            # Generate random ordering of the sets
            order = np.arange(len(files))
            np.random.shuffle(order)

            # Create split mapping
            index = 0
            split = {}
            for i in range(len(sets)):
                cur = 0
                while cur < samples[i]:
                    split[ files[order[index]] ] = sets[i]
                    cur = cur + 1
                    index = index + 1

        # Create set folders
        for s in sets:
            os.mkdir( os.path.join(data_path, s) )

        # Divide files into groups
        for f in split:
            shutil.move(os.path.join(data_path, f), os.path.join( os.path.join(data_path, split[f]), f ))
        return split

    def generator(self, dataset, batch_size, crop_offset=True, normalize=False, params={}):
        path = os.path.join( os.path.join(self.path, self.conf['paths']['data']), dataset )

        # Get all files & classes
        x_set, y_set = [], []
        for f in os.listdir(path):
            x_set.append(os.path.join(path, f))
            arr = f.split('_')
            y_set.append(int(arr[-1][:-5]))

        # Shuffle data
        length = len(x_set)
        indices = np.arange(length)
        np.random.shuffle(indices)
        x_set = np.asarray(x_set)[indices]
        y_set = np.asarray(y_set)[indices]

        if crop_offset:
            x_set = x_set[ :(length // batch_size) * batch_size ]
            y_set = y_set[ :(length // batch_size) * batch_size ]

        # Construct sequence out of filtered sets
        return ImageSequence(x_set, to_categorical(y_set, self.nb_classes), self.input_shape, batch_size, normalize=normalize)

