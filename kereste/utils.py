import os
import json

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from keras.models import model_from_json

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import ProjectedGradientDescent, FastGradientMethod

from .models.vgg import build_vgg
from .datasets import smallNORB

def init_model(model_conf, input_shape, nb_classes):
    if 'class' in model_conf:
        if model_conf['class'] == 'vgg':            
            name = model_conf['name']
            batch_size = model_conf['batch_size']
            group_size = model_conf['group_size']
            norm_type = model_conf['norm_type']
            return build_vgg(name=name, input_shape=input_shape, norm_type=norm_type, batch_size=batch_size, group_size=group_size, nb_classes=nb_classes)
    
    raise('Model is not supported')
    
def init_dataset(dataset, path):
    if dataset == 'smallNORB':
        return smallNORB.from_json(path)
    
    raise('Dataset is not supported')
    
def init_attack(model, attack):
    wrap = KerasModelWrapper(model)
    sess = K.get_session()
    if attack == 'pgd':
        return ProjectedGradientDescent(wrap, sess=sess)
    elif attack == 'fgm':
        return FastGradientMethod(wrap, sess=sess)
    
    raise('Adversarial attack is not supported')