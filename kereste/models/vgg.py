# VGG with no dropout applied!

from keras.models import Model
from keras.layers import Input, Conv2D, Activation, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras.regularizers import l2
from .localnorm import LocalNormalization

def norm(norm_type, batch_size=None, groupsize=None):
    if norm_type=='batch':
        return BatchNormalization()
    elif norm_type=='local':
        return LocalNormalization(axis=0, batch_size=batch_size, groupsize=groupsize)
    return None

def build_vgg(name='vgg', input_shape=(96,96,1,), input_layer=None, nb_classes=5, weight_decay=0.0005, norm_type='batch', batch_size=128, group_size=8, **kwargs):
    if input_layer is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = input_layer
    x = inputs
  
    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(64, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(128, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(256, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)


    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = Conv2D(512, (3, 3), padding='same',kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    if norm_type is not None:
        x = norm(norm_type, batch_size=batch_size, groupsize=group_size)(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(512,kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)

    x = Dropout(0.5)(x)
    x = Dense(nb_classes)(x)
    x = Activation('softmax', name=name+'_output')(x)
  
    outputs = x
    return Model(name=name, inputs=inputs, outputs=outputs)