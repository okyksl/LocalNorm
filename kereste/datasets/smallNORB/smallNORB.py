import os
import json
import shutil
import numpy as np

from keras.utils import to_categorical

from ..dataset import Dataset
from ..generators import ImageSequence
from .small_norb.smallnorb.dataset import SmallNORBDataset

class smallNORB(Dataset):
    nb_classes = 5
    class_labels = ['animal', 'human', 'airplane', 'truck', 'car']
    input_shape = (96, 96, 1)

    @staticmethod
    def download(download_path, download_dependency=True):
        if download_dependency:
            script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dependency.sh')
            dependency_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'small_norb')
            os.system('sh %s %s' % (script_path, dependency_path))
            
        script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'download.sh')
        os.system('sh %s %s' % (script_path, download_path))

    @staticmethod
    def init(data_path, download_path):
        os.mkdir( data_path )
        dataset = SmallNORBDataset(dataset_root=download_path)
        dataset.export_to_jpg(data_path)

        test_path = os.path.join(data_path, 'test')
        files = os.listdir(test_path)
        for f in files:
            shutil.move(os.path.join(test_path, f), data_path)
        os.rmdir(test_path)

        train_path = os.path.join(data_path, 'train')
        files = os.listdir(train_path)
        for f in files:
            shutil.move(os.path.join(train_path, f), data_path)
        os.rmdir(train_path)

    @staticmethod
    def split(data_path, conf, split=None):
        sets = conf['sets']
        instances = 10

        if split is None:
            ratios = conf['ratios']
            ratios = np.asarray(ratios) * instances

            # Generate random ordering of the sets
            order = np.arange(instances)
            np.random.shuffle(order)

            # Create split mapping
            index = 0
            split = {}
            for i in range(len(sets)):
                cur = 0
                while cur < ratios[i]:
                    """
                    # Add all files in the group to the set
                    for f in groups[order[index]]:
                        split[f] = sets[i]
                    """
                    split[str(int(order[index]))] = sets[i]
                    cur = cur + 1
                    index = index + 1

        # Create set folders
        for s in sets:
            os.mkdir( os.path.join(data_path, s) )

        # Divide files into groups
        files = os.listdir(data_path)
        for f in files:
            for i in range(instances):
                if ('%02di' % i) in f:
                    shutil.move(os.path.join(data_path, f), os.path.join( os.path.join(data_path, split[str(i)]), f ))
                    break
        return split

    @staticmethod
    def process(path):
        with open(path) as f:
            conf = json.load(f)
        dataset_dir = os.path.dirname(path)

        download_path = os.path.join(dataset_dir, conf['paths']['download'])
        if conf['status'] == 'download':
            smallNORB.download(download_path)
            conf['status'] = 'init'

        data_path = os.path.join(dataset_dir, conf['paths']['data'])
        if conf['status'] == 'init':
            smallNORB.init(data_path, download_path)
            conf['status'] = 'split'

        split_path = os.path.join(dataset_dir, conf['paths']['split'])
        if conf['status'] == 'split':
            split = None
            if os.path.isfile(split_path):
                with open(split_path) as f:
                    split = json.load(f)

            split = smallNORB.split(data_path, conf['split'], split)
            smallNORB.to_json(split_path, split)
            conf['status'] ='ready'

        if conf['status'] == 'ready':
            smallNORB.to_json(path, conf)

        return smallNORB(path, conf)

    def generator(self, dataset, batch_size, crop_offset=True, normalize=False, params={}):
        path = os.path.join( os.path.join(self.path, self.conf['paths']['data']), dataset )

        x_set = []
        y_set = []
        attrs = ['category', 'instance', 'lighting', 'elevation', 'azimuth']
        for f in os.listdir(path):
            # Read Sample
            arr = f.split('_')
            sample = {}
            for i in range(len(attrs)):
                sample[attrs[i]] = int(arr[i][:2])

            # Filter Sample
            if 'lighting' in params and sample['lighting'] not in params['lighting']:
                continue
            if 'elevation' in params and sample['elevation'] not in params['elevation']:
                continue
            if 'azimuth' in params and sample['azimuth'] not in params['azimuth']:
                continue

            if 'categories' in params and sample['category'] in params['categories']:
                if sample['lighting'] not in params.categories[sample['category']]['lighting']:
                    continue
                if sample['elevation'] not in params.categories[sample['category']]['elevation']:
                    continue
                if sample['azimuth'] not in params.categories[sample['category']]['azimuth']:
                    continue

            x_set.append(os.path.join(path, f))
            y_set.append(sample['category'])

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
