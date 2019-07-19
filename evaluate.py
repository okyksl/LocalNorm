import numpy as np

from keras import backend as K
from keras.metrics import categorical_accuracy

def vote(preds, classes):
    guesses = np.argmax(preds, axis=-1)
    result = np.zeros(classes)
    for i in guesses:
        result[i] = result[i] + 1
    return result * 1./len(guesses)

def evaluate_preds(true_y, preds_y):
    true_y = K.variable(true_y)
    preds_y = K.variable(preds_y)
    return K.eval(K.mean(categorical_accuracy(true_y, preds_y)))

def evaluate_model(model, test_data, batch_size=32, group_size=8, eval_type='batch', group=0):
    test_x, test_y = test_data
    classes = test_y.shape[-1]
    
    if eval_type == 'single':
        preds_y = []
        for i in range(len(test_x)):
            input_x = np.repeat(test_x[i][np.newaxis, ...], batch_size, axis=0)
            preds = model.predict(input_x, batch_size=batch_size)
            preds_y.append( preds[group * group_size] )
            
        preds_y = np.stack(preds_y)
        return evaluate_preds(test_y, preds_y)
    
    elif eval_type == 'single-voting':
        preds_y = []
        for i in range(len(test_x)):
            input_x = np.repeat(test_x[i][np.newaxis, ...], batch_size, axis=0)
            preds = model.predict(input_x, batch_size=batch_size)
            preds_y.append( vote(preds, classes) )
            
        preds_y = np.stack(preds_y)
        return evaluate_preds(test_y, preds_y)
        
    elif eval_type == 'voting':
        preds_y = []
        for i in range(len(test_x) // group_size):
            input_x = np.tile(test_x[i*group_size:(i+1)*group_size], (batch_size // group_size, 1, 1, 1))
            preds = model.predict(input_x, batch_size=batch_size)
            for j in range(group_size):
                preds_y.append( vote(preds[j::group_size], classes) )
            
        preds_y = np.stack(preds_y)
        return evaluate_preds(test_y, preds_y)
    
    else: # 'batch' - default
        preds_y = model.predict(test_x, batch_size=batch_size)
        return evaluate_preds(test_y, preds_y)