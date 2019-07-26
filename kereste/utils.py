import os
import json

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from keras.models import model_from_json

import cleverhans.attacks
from cleverhans.utils_keras import KerasModelWrapper

from kereste.models.vgg import build_vgg
import kereste.datasets as datasets

def init_model(model_conf, input_shape, nb_classes):
    if 'class' in model_conf:
        if model_conf['class'] == 'vgg':
            return build_vgg(input_shape=input_shape, nb_classes=nb_classes, **model_conf)
    
    raise('Model is not supported')
    
def init_dataset(dataset, path):
    return getattr(datasets, dataset).from_json(path)
    
def init_attack(model, attack):
    wrap = KerasModelWrapper(model)
    sess = K.get_session()
    attack = getattr(cleverhans.attacks, attack)
    return attack(wrap, sess)
