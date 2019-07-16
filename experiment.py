import os
import json

from numpy import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD

from utils import init_dataset, init_models, init_attack

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
            self.status = self.conf['status']
            self.dataset = init_dataset(self.conf['dataset'])
            self.models = init_models(self.conf['model'], self.conf['dataset'])
            
    # Save experiment configurations to a directory
    def save(self, path=None):
        if path is None:
            path = self.path

        with open(path, 'w') as f:
            json.dump(self.conf, f)
    
    # Prepare data according to configurations
    def prepare(self, dataset, data_conf):
        # Extract params
        if 'params' in data_conf:
            data_params = data_conf['params']
        else:
            data_params = {}
        
        # Get data
        data = self.dataset.process(dataset, data_params)

        # Clip overflow
        batch_size = self.conf['model']['batch_size']
        data_len = (len(data[0]) // batch_size) * batch_size
        data = ( data[0][:data_len], data[1][:data_len] )
        
        # Apply adversarial attack
        if 'adversarial' in data_conf:
            adv_conf = data_conf['adversarial']
            if 'params' in adv_conf:
                adv_params = adv_conf['params']
            else:
                adv_params = {}
                
            attack_batch = init_attack(self.models[0], adv_conf['attack'])
            attack_local = init_attack(self.models[1], adv_conf['attack'])

            data_batch_x = []
            data_local_x = []
            for i in range( data_len // batch_size ):
                batch_x = data[0][i*batch_size:(i+1)*batch_size]
                data_batch_x.append( attack_batch.generate_np(batch_x, **adv_params) )
                data_local_x.append( attack_local.generate_np(batch_x, **adv_params) )
                
            data_batch_x = concatenate(data_batch_x)
            data_local_x = concatenate(data_local_x)
            return (data_batch_x, data[1]), (data_local_x, data[1])
        else:
            return data, data
    
    # Preprocess before training/evaluating
    def preprocess(self):
        # Split dataset
        data_split = self.conf['train_conf']['data_split']
        self.dataset.split(data_split)
        
    # Train the model according to configurations
    def train(self):
        if self.conf['status'] != 'train':
            raise('The model is already trained.')
        
        # Models
        model_batch = self.models[0]
        model_local = self.models[1]

        # Hyperparams
        batch_size = self.conf['model']['batch_size']
        train_conf = self.conf['train_conf']
        data_conf = train_conf['data_conf']
        epochs = train_conf['epochs']
        learning_rate = train_conf['learning_rate']
        
        # Data
        train_batch, train_local = self.prepare('train', data_conf)
        val_batch, val_local = self.prepare('val', data_conf)

        # Checkpoints
        checkpoint_batch = ModelCheckpoint( os.path.join(self.directory, 'weights-batch.h5'), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        checkpoint_local = ModelCheckpoint( os.path.join(self.directory, 'weights-local.h5'), monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        # Optimizers
        optimizer_batch = SGD(lr=learning_rate, momentum=0.9)
        optimizer_local = SGD(lr=learning_rate, momentum=0.9)

        # Compile
        model_batch.compile(loss='categorical_crossentropy', optimizer=optimizer_batch, metrics=['accuracy'])
        model_local.compile(loss='categorical_crossentropy', optimizer=optimizer_local, metrics=['accuracy'])

        # Train
        hist_batch = model_batch.fit(x=train_batch[0], y=train_batch[1], batch_size=batch_size, epochs=epochs, validation_data=val_batch, callbacks=[checkpoint_batch])
        hist_local = model_local.fit(x=train_local[0], y=train_local[1], batch_size=batch_size, epochs=epochs, validation_data=val_local, callbacks=[checkpoint_local])
        
        # Save Models
        with open( os.path.join(self.directory, 'model-batch.json'), 'w') as f:
            f.write(model_batch.to_json())
        with open( os.path.join(self.directory, 'model-local.json'), 'w') as f:
            f.write(model_local.to_json())
        
        # Save Results
        self.conf['status'] = 'test'
        self.conf['results']['batch'] = {
            'train': hist_batch.history
        }
        self.conf['results']['local'] = {
            'train': hist_local.history
        }
        self.conf['model']['weights'] = {
            'batch': os.path.join(self.directory, 'weights-batch.h5'),
            'local': os.path.join(self.directory, 'weights-local.h5')
        }
        self.conf['model']['path'] = {
            'batch': os.path.join(self.directory, 'model-batch.json'),
            'local': os.path.join(self.directory, 'model-local.json')
        }
       
    # Executes an experiment stated in configuration file
    def execute(self, experiment=None, experiment_conf=None):
        # Execute all experiments if nothing is specified
        if experiment is None:
            for exp in self.conf['experiments']:
                self.execute(exp)
            return
        
        if experiment_conf is not None:
            self.conf['experiments'][experiment] = experiment_conf
        
        # Hyperparams
        batch_size = self.conf['model']['batch_size']
        
        # Data
        experiment_conf = self.conf['experiments'][experiment]
        test_batch, test_local = self.prepare('test', experiment_conf)

        # Models
        model_batch = self.models[0]
        model_local = self.models[1]

        # Evaluate
        result_batch = model_batch.evaluate(x=test_batch[0], y=test_batch[1], batch_size=batch_size)
        result_local = model_local.evaluate(x=test_local[0], y=test_local[1], batch_size=batch_size)

        # Results
        self.conf['results']['batch'][experiment] = result_batch
        self.conf['results']['local'][experiment] = result_local