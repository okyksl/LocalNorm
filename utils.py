import os
import json

from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
from keras.models import model_from_json

from models.vgg import build_vgg
from datasets.smallNORB import smallNORB

from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import ProjectedGradientDescent, FastGradientMethod

def init_model(model_conf, dataset_conf):
    if 'path' in model_conf:
        model = model_from_json(model_conf['path'])
        
        if 'weights' in model_conf:
            model.load_weights(model_conf['weights'])
        return model
    
    if 'class' in model_conf:
        if model_conf['class'] == 'vgg':
            name = dataset_conf['name']
            nb_classes = dataset_conf['nb_classes']
            input_shape = dataset_conf['input_shape']
            batch_size = model_conf['batch_size']
            group_size = model_conf['group_size']
            norm_type = model_conf['norm_type']
            return build_vgg(name=name, input_shape=input_shape, norm_type=norm_type, batch_size=batch_size, group_size=group_size, nb_classes=nb_classes)
    
    raise('Model is not supported')
    
def init_dataset(dataset_conf):
    if dataset_conf['name'] == 'smallNORB':
        return smallNORB()
    
    raise('Dataset is not supported')
    
def init_attack(model, attack):
    wrap = KerasModelWrapper(model)
    sess = K.get_session()
    if attack == 'pgd':
        return ProjectedGradientDescent(wrap, sess=sess)
    elif attack == 'fgm':
        return FastGradientMethod(wrap, sess=sess)
    
    raise('Adversarial attack is not supported')