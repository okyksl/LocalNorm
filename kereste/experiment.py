import os
import json

import numpy as np
import keras.backend as K

from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LambdaCallback

from .evaluate import evaluate_model
from .utils import init_dataset, init_model, init_attack
from .datasets.generators import AugmenterSequence

class Experiment:
    def __init__(self, path=None):
        if path is not None:
            self.load(path)
        
    # Load experiment configurations from a directory
    def load(self, path):
        self.path = path
        self.directory = os.path.dirname(path)
        with open(path) as f:
            self.conf = json.load(f)
            
            self.name = self.conf['name']
            self.dataset = init_dataset(self.conf['dataset']['class'], self.conf['dataset']['path'])
            self.models = {}
            self.epochs = {}
            for model in self.conf['models']:
                self.models[model] = init_model(self.conf['models'][model], self.dataset.input_shape, self.dataset.nb_classes)
                self.epochs[model] = 0

    # Save experiment configurations to a directory
    def save(self, path=None):
        if path is None:
            path = self.path

        with open(path, 'w') as f:
            json.dump(self.conf, f, indent=4)
    
    # Restore models from desired batch
    def restore(self, model=None, epoch=0):
        if model is None:
            for model in self.models:
                self.restore(model=model, epoch=epoch)
            return
        
        if epoch == 0: # Get best result
            score = 0
            for file in os.listdir(self.directory):
                if file.startswith(model) and file.endswith('.h5'):
                    sc = float(file[-9:-3])
                    if sc > score:
                        epoch = int(file[-12:-10])
                        score = sc

        prefix = '%s_%02d' % (model, epoch)
        for file in os.listdir(self.directory):
            if file.startswith(prefix) and file.endswith('.h5'):
                self.models[model].load_weights(os.path.join(self.directory, file))
                self.register(model, epoch)
                break
    
    # Register model epoch
    def register(self, model, epoch):
        self.epochs[model] = epoch

    # Prepare data according to configurations
    def prepare(self, model, dataset, data_conf):
        # Extract params
        if 'params' in data_conf:
            data_params = data_conf['params']
        else:
            data_params = {}
        
        # Get data
        batch_size = self.conf['models'][model]['batch_size']
        generator = self.dataset.generator(dataset, batch_size, params=data_params)
        
        # Apply adversarial attack
        if 'adversarial' in data_conf:            
            adv_conf = data_conf['adversarial']
            if 'params' in adv_conf:
                adv_params = adv_conf['params']
            else:
                adv_params = {}
            if 'model' in adv_conf:
                adv_model = adv_conf['model']
            else:
                adv_model = model

            attack = init_attack(self.models[adv_model], adv_conf['attack'])
            
            def gen_augmenter(attack, params):
                def adv_augment(x):
                    return attack.generate_np(x, **params)
                return adv_augment
            
            return AugmenterSequence(generator, gen_augmenter(attack, adv_params))
        else:
            return generator
    
    # Preprocess before training/evaluating
    def preprocess(self):
        # Process if dataset needs to be processed
        self.dataset.process(self.conf['dataset']['path'])
        
    # Train the model according to configurations
    def train(self, model=None, exec_every_epoch=False):
        if model is None:
            for model in self.models:
                self.train(model=model, exec_every_epoch=exec_every_epoch)
            return
        
        # Hyperparams
        train_conf = self.conf['training']
        data_conf = train_conf['data']
        
        batch_size = self.conf['models'][model]['batch_size']
        group_size = self.conf['models'][model]['group_size']
        epochs = train_conf['epochs']
        learning_rate = train_conf['learning_rate']
        
        # Data
        train_gen = self.prepare(model=model, dataset='train', data_conf=data_conf)        
        val_gen = self.prepare(model=model, dataset='val', data_conf=data_conf)

        # Callbacks
        output_file = os.path.join(self.directory, model + '_{epoch:02d}_{val_acc:.4f}.h5')
        checkpoint = ModelCheckpoint(output_file, monitor='val_acc', period=1, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20, restore_best_weights=True, verbose=1)
        lr_regularizer = ReduceLROnPlateau(monitor='val_loss', min_delta=0.001, factor=0.5, patience=5, verbose=1)
        epoch_callback = LambdaCallback(
            on_epoch_end=lambda epoch,logs: self.register(model, epoch)) # update epoch
        callbacks = [ checkpoint, early_stopping, lr_regularizer, epoch_callback ]
        
        # Experiment Callback
        if exec_every_epoch:
            experiment_callback = LambdaCallback(
                on_epoch_end=lambda epoch,logs: self.execute(model=model))
            callbacks = callbacks + [ experiment_callback ]
        
        # Compile
        optimizer = SGD(lr=learning_rate, momentum=0.9)
        self.models[model].compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
        # Train
        hist = self.models[model].fit_generator(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks
        )
        self.register(model, early_stopping.stopped_epoch) # restore best epoch
        
        # Save Model Def
        with open( os.path.join(self.directory, 'model-' + model + '.json'), 'w') as f:
            f.write(self.models[model].to_json())

        # Save Results
        history = hist.history
        if 'lr' in history:
            for i in range(len(history['lr'])):
                history['lr'][i] = float(history['lr'][i]) # Convert numpy float to regular float

        if 'training_history' not in self.conf:
            self.conf['training_history'] = {}
        self.conf['training_history'][model] = history
        
    # Executes an experiment stated in configuration file
    def execute(self, model=None, experiment=None, experiment_conf=None):
        # Execute on all models if model is not specified
        if model is None:
            for model in self.models:
                self.execute(model=model, experiment=experiment, experiment_conf=experiment_conf)
            return

        # Execute all experiments if experiment is not specified
        if experiment is None:
            for experiment in self.conf['experiments']:
                self.execute(model=model, experiment=experiment)
            return
        
        # Get experiment config if not specified
        if experiment_conf is not None:
            self.conf['experiments'][experiment] = experiment_conf

        # Hyperparams
        batch_size = self.conf['models'][model]['batch_size']
        group_size = self.conf['models'][model]['group_size']

        # Data
        experiment_conf = self.conf['experiments'][experiment]        
        exp_gen = self.prepare(model=model, dataset='test', data_conf=experiment_conf)
        
        # Evaluate
        eval_types = self.conf['models'][model]['eval']
        results = {}
        for eval_type in eval_types:
            print('Model %s, Experiment %s, Eval %s is executing...' % (model, experiment, eval_type))
            results[eval_type] = evaluate_model(self.models[model],
                                                exp_gen,
                                                batch_size=batch_size,
                                                group_size=group_size,
                                                eval_type=eval_type)
            print(results[eval_type])

        # Save Results
        if 'results' not in self.conf:
            self.conf['results'] = {}
        if experiment not in self.conf['results']:
            self.conf['results'][experiment] = {}
        if model not in self.conf['results'][experiment]:
            self.conf['results'][experiment][model] = {}
        for res in results:
            if res not in self.conf['results'][experiment][model]:
                self.conf['results'][experiment][model][res] = {}
            self.conf['results'][experiment][model][res][str(self.epochs[model])] = float(results[res])
        
        # Print Results
        print('Model %s, Experiment %s results:' % (model, experiment))
        print(results)
        
    def run(self, exec_every_epoch=False):
        self.preprocess()
        self.train(exec_every_epoch=exec_every_epoch)
        if not exec_every_epoch:
            self.execute()