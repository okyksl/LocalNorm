import os
import json

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from keras.models import model_from_json

from models.vgg import build_vgg
from datasets.smallNORB import smallNORB

def init_models(model_conf, dataset_conf):
    if 'path' in model_conf:
        model_batch = model_from_json(model_conf['path']['batch'])
        model_local = model_from_json(model_conf['path']['local'])
        
        if 'weights' in model_conf:
            model_batch.load_weights(model_conf['weights']['batch'])
            model_local.load_weights(model_conf['weights']['local'])
        return model_batch, model_local

    if 'name' in model_conf:
        if model_conf['name'] == 'vgg':
            nb_classes = dataset_conf['nb_classes']
            input_shape = dataset_conf['input_shape']
            batch_size = model_conf['batch_size']
            group_size = model_conf['group_size']

            model_batch = build_vgg(name='batch_vgg', input_shape=input_shape, norm_type='batch', batch_size=batch_size, group_size=group_size, nb_classes=nb_classes)
            model_local = build_vgg(name='local_vgg', input_shape=input_shape, norm_type='local', batch_size=batch_size, group_size=group_size, nb_classes=nb_classes)
            return model_batch, model_local
    
    raise('Model is not supported')
    
def init_dataset(dataset_conf):
    if dataset_conf['name'] == 'smallNORB':
        return smallNORB()
    raise('Dataset is not supported')
