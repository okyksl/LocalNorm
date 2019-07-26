import os
import json

from abc import ABC, abstractmethod
    
class Dataset(ABC):
    @property
    def nb_classes(self):
        raise NotImplementedError

    @property
    def class_labels(self):
        raise NotImplementedError
    
    @property
    def input_shape(self):
        raise NotImplementedError
    
    def __init__(self, path, conf):
        self.json_path = path
        self.path = os.path.dirname(path)
        self.conf = conf

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            conf = json.load(f)
        return cls(path, conf)
    
    @classmethod
    def to_json(cls, path, conf):
        with open(path, 'w') as f:
            json.dump(conf, f, indent=4)
    
    def save(self):
        self.to_json(self.path, self.conf)
    
    @staticmethod
    @abstractmethod
    def download(download_path):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def init(data_path, download_path):
        raise NotImplementedError
    
    @staticmethod
    @abstractmethod
    def split(data_path, conf, split=None):
        raise NotImplementedError
    
    @classmethod
    def process(cls, path):
        with open(path) as f:
            conf = json.load(f)
        dataset_dir = os.path.dirname(path)

        download_path = os.path.join(dataset_dir, conf['paths']['download'])
        if conf['status'] == 'download':
            cls.download(download_path)
            conf['status'] = 'init'

        data_path = os.path.join(dataset_dir, conf['paths']['data'])
        if conf['status'] == 'init':
            cls.init(data_path, download_path)
            conf['status'] = 'split'

        split_path = os.path.join(dataset_dir, conf['paths']['split'])
        if conf['status'] == 'split':
            split = None
            if os.path.isfile(split_path):
                with open(split_path) as f:
                    split = json.load(f)

            split = cls.split(data_path, conf['split'], split)
            cls.to_json(split_path, split)
            conf['status'] ='ready'

        if conf['status'] == 'ready':
            cls.to_json(path, conf)

        return cls(path, conf)

    @abstractmethod
    def generator(self, dataset, batch_size, crop_offset=True, normalize=False, params={}):
        raise NotImplementedError