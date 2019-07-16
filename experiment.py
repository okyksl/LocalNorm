import os
import json

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD

from utils import init_dataset, init_models

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
    
    # Preprocess dataset
    def preprocess(self):
        self.dataset.split(self.conf['train_conf']['data_split'])
        self.train_data = self.dataset.process('train', self.conf['train_conf']['data_conf'])
        self.val_data = self.dataset.process('val', self.conf['train_conf']['data_conf'])
        
        # Clip batches
        batch_size = self.conf['model']['batch_size']
        train_len = (len(self.train_data[0]) // batch_size) * batch_size
        val_len = (len(self.val_data[0]) // batch_size) * batch_size

        self.train_data = ( self.train_data[0][:train_len], self.train_data[1][:train_len] )
        self.val_data = ( self.val_data[0][:val_len], self.val_data[1][:val_len] )
        
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
        epochs = train_conf['epochs']
        learning_rate = train_conf['learning_rate']
        
        # Data
        train_x, train_y = self.train_data
        val_data = self.val_data

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
        hist_batch = model_batch.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=val_data, callbacks=[checkpoint_batch])
        hist_local = model_local.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, validation_data=val_data, callbacks=[checkpoint_local])
        
        # Save Models
        with open( os.path.join(self.directory, 'model-batch.json'), 'w') as f:
            f.write(model_batch.to_json())
        with open( os.path.join(self.directory, 'model-local.json'), 'w') as f:
            f.write(model_local.to_json())
        
        # Results
        self.conf['results']['batch']['train'] = hist_batch.history
        self.conf['results']['local']['train'] = hist_local.history
        self.conf['status'] = 'test'
        self.conf['model']['weights'] = {
            'batch': os.path.join(self.directory, 'weights-batch.h5'),
            'local': os.path.join(self.directory, 'weights-local.h5')
        }
        self.conf['model']['path'] = {
            'batch': os.path.join(self.directory, 'model-batch.json'),
            'local': os.path.join(self.directory, 'model-local.json')
        }
       
    # Executes an experiment stated in configuration file
    def execute(self, experiment=None):
        # Execute all experiments if nothing is specified
        if experiment is None:
            for exp in self.conf['experiments']:
                self.execute(exp)
            return
        
        # Data
        experiment_conf = self.conf['experiments'][experiment]
        test_x, test_y = self.dataset.process('test', experiment_conf)

        # Clip Batches
        batch_size = self.conf['model']['batch_size']
        test_len = (len(test_x) // batch_size) * batch_size

        test_x = test_x[:test_len]
        test_y = test_y[:test_len]

        # Models
        model_batch = self.models[0]
        model_local = self.models[1]

        # Evaluate
        result_batch = model_batch.evaluate(x=test_x, y=test_y, batch_size=batch_size)
        result_local = model_local.evaluate(x=test_x, y=test_y, batch_size=batch_size)

        # Results
        self.conf['results']['batch'][experiment] = result_batch
        self.conf['results']['local'][experiment] = result_local