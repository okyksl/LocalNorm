import os
import random
import numpy as np
from keras.utils import to_categorical
from .small_norb.smallnorb.dataset import SmallNORBDataset

class smallNORB:
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            dataset_root = os.path.dirname(os.path.abspath(__file__))
        else:
            dataset_root = dataset_dir
        
        self.dataset = SmallNORBDataset(dataset_root=dataset_root)
        self.categories = ['animal', 'human', 'airplane', 'truck', 'car']
        self.nb_classes = 5
        self.input_shape = (96, 96, 1)
        
    def split(self, conf):
        # Get original test and train sets
        test_set = self.dataset.group_dataset_by_category_and_instance('test')
        train_set = self.dataset.group_dataset_by_category_and_instance('train')
        
        # shuffle dataset
        dataset = test_set + train_set
        random.shuffle(dataset)
        
        # class based sets
        train_set = [ [] for i in range(len(self.categories)) ]
        val_set = [ [] for i in range(len(self.categories)) ]
        test_set = [ [] for i in range(len(self.categories)) ]
        
        for toy_set in dataset:
            cat = toy_set[0].category
            if len(val_set[cat]) < conf['val']:
                val_set[cat].append(toy_set)
            elif len(test_set[cat]) < conf['test']:
                test_set[cat].append(toy_set)
            else:
                train_set[cat].append(toy_set)
                
        self.train_set  = [toy_set for cat in train_set for toy_set in cat]
        self.val_set  = [toy_set for cat in val_set for toy_set in cat]
        self.test_set  = [toy_set for cat in test_set for toy_set in cat]

    def process(self, dataset, conf):
        if dataset == 'train':
            dataset = self.train_set
        elif dataset == 'val':
            dataset = self.val_set
        elif dataset == 'test':
            dataset = self.test_set
        else:
            raise('Dataset should be either train, val or test.')
        
        x_values, y_values = [], []
        for toy_set in dataset:
            for sample in toy_set:
                if 'elevation' in conf and sample.elevation not in conf['elevation']:
                    continue
                if 'azimuth' in conf and sample.azimuth not in conf['azimuth']:
                    continue
                if 'lighting' in conf and sample.lighting not in conf['lighting']:
                    continue
                
                if 'categories' in conf and sample.category in conf['categories']:
                    if sample.elevation not in conf.categories[sample.category]['elevation']:
                        continue
                    if sample.azimuth not in conf.categories[sample.category]['azimuth']:
                        continue
                    if sample.lighting not in conf.categories[sample.category]['lighting']:
                        continue
                    
                x_values.append( np.expand_dims(sample.image_lt, axis=-1) )
                y_values.append( to_categorical(sample.category, len(self.categories) ) )
                
        x_values = np.asarray(x_values)
        y_values = np.asarray(y_values)
        np.random.shuffle(x_values)
        np.random.shuffle(y_values)
        return x_values, y_values
