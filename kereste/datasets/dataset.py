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
    
    @staticmethod
    @abstractmethod
    def process(path):
        raise NotImplementedError
        
    @abstractmethod
    def generator(self, dataset, batch_size, crop_offset=True, params={}):
        raise NotImplementedError