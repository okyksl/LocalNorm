import os
import json

import numpy as np
import keras.backend as K

from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import SGD

from evaluate import evaluate_model
from utils import init_dataset, init_model, init_attack

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
            self.dataset = init_dataset(self.conf['dataset'])
            self.models = {}
            for model in self.conf['models']:
                self.models[model] = init_model(self.conf['models'][model], self.conf['dataset'])
            
    # Save experiment configurations to a directory
    def save(self, path=None):
        if path is None:
            path = self.path

        with open(path, 'w') as f:
            json.dump(self.conf, f)
    
    # Prepare data according to configurations
    def prepare(self, model, dataset, data_conf):
        # Extract params
        if 'params' in data_conf:
            data_params = data_conf['params']
        else:
            data_params = {}
        
        # Get data
        data = self.dataset.process(dataset, data_params)

        # Clip overflow
        batch_size = self.conf['models'][model]['batch_size']
        data_len = (len(data[0]) // batch_size) * batch_size
        data = ( data[0][:data_len], data[1][:data_len] )
        
        # Shuffle data
        indices = np.arange(data_len)
        np.random.shuffle(indices)
        data = ( data[0][indices], data[1][indices] )
        
        # Apply adversarial attack
        if 'adversarial' in data_conf:
            adv_conf = data_conf['adversarial']
            if 'params' in adv_conf:
                adv_params = adv_conf['params']
            else:
                adv_params = {}
                
            attack = init_attack(self.models[model], adv_conf['attack'])
            
            # Generate adversarial samples batch by batch
            data_x = []
            for i in range( data_len // batch_size ):
                batch_x = data[0][i*batch_size:(i+1)*batch_size]
                data_x.append( attack.generate_np(batch_x, **adv_params) )
                
            data_x = np.concatenate(data_x)
            return (data_x, data[1])
        else:
            return data
    
    # Preprocess before training/evaluating
    def preprocess(self):
        # Split dataset
        data_split = self.conf['train_conf']['data_split']
        self.dataset.split(data_split)
        
    # Train the model according to configurations
    def train(self, model=None, exec_every_epoch=False):
        if model is None:
            for model in self.models:
                self.train(model=model, exec_every_epoch=exec_every_epoch)
            return
        
        # Hyperparams
        train_conf = self.conf['train_conf']
        data_conf = train_conf['data_conf']
        
        batch_size = self.conf['models'][model]['batch_size']
        group_size = self.conf['models'][model]['group_size']
        epochs = train_conf['epochs']
        learning_rate = train_conf['learning_rate']
        
        # Data
        train_x, train_y = self.prepare(model=model, dataset='train', data_conf=data_conf)        
        val_data = self.prepare(model=model, dataset='val', data_conf=data_conf)

        # Checkpoint
        checkpoint = ModelCheckpoint( os.path.join(self.directory, 'weights-' + model + '.h5'), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        callbacks = [ checkpoint ]
        
        # Experiment Callback
        if exec_every_epoch:
            experiment_callback = LambdaCallback(
                on_epoch_end=lambda epoch,logs: self.execute(model=model))
            callbacks = callbacks + [ experiment_callback ]
        
        # Compile
        optimizer = SGD(lr=learning_rate, momentum=0.9)
        self.models[model].compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
        # Train
        hist = self.models[model].fit(
            x=train_x,
            y=train_y,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=val_data,
            callbacks=callbacks
        )
        
        # Save Model Def
        with open( os.path.join(self.directory, 'model-' + model + '.json'), 'w') as f:
            f.write(self.models[model].to_json())
            
        # Save Weight Path
        self.conf['models'][model]['weights'] = os.path.join(self.directory, 'weights-' + model + '.h5')
        
        # Save Model Def Path
        self.conf['models'][model]['path'] = os.path.join(self.directory, 'model-' + model + '.h5')
        
        # Save Results
        if 'results' not in self.conf:
            self.conf['results'] = {}
        if 'train' not in self.conf['results']:
            self.conf['results']['train'] = {}
        self.conf['results']['train'][model] = hist.history
       
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
       
        # Print Experiment Start
        print('Model %s, Experiment %s is executing...' % (model, experiment))
        
        # Hyperparams
        batch_size = self.conf['models'][model]['batch_size']
        group_size = self.conf['models'][model]['group_size']

        # Data
        experiment_conf = self.conf['experiments'][experiment]        
        test_data = self.prepare(model=model, dataset='test', data_conf=experiment_conf)
        
        # Evaluate
        eval_types = self.conf['models'][model]['eval']
        results = {}
        for eval_type in eval_types:
            print('Model %s, Experiment %s, Eval %s is executing...' % (model, experiment, eval_type))
            results[eval_type] = evaluate_model(self.models[model],
                                                test_data=test_data,
                                                batch_size=batch_size,
                                                group_size=group_size,
                                                eval_type=eval_type)
            print('Model %s, Experiment %s, Eval %s results: ' % (model, experiment, eval_type))
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
                self.conf['results'][experiment][model][res] = { 'acc': [] }
            self.conf['results'][experiment][model][res]['acc'].append(float(results[res]))
        
        # Print Results
        print('Model %s, Experiment %s results:' % (model, experiment))
        print(results)
        
    def run(self):
        self.preprocess()
        self.train(exec_every_epoch=True)
