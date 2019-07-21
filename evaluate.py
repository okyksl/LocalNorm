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

def evaluate_model(model, generator, batch_size=32, group_size=8, eval_type='batch', group=0):
    true_y = []
    preds_y = []
    iterator = iter(generator)

    if eval_type == 'single':
        for i in range(len(generator)):
            batch_x, batch_y = next(iterator)
            true_y.append(batch_y)

            for j in range(len(batch_x)):
                input_x = np.repeat(batch_x[j][np.newaxis, ...], batch_size, axis=0)
                preds = model.predict(input_x, batch_size=batch_size)
                preds_y.append( preds[group * group_size] )
            
        true_y = np.concatenate(true_y)
        preds_y = np.stack(preds_y)
        return evaluate_preds(true_y, preds_y)
    
    elif eval_type == 'single-voting':
        for i in range(len(generator)):
            batch_x, batch_y = next(iterator)
            true_y.append(batch_y)

            for j in range(len(batch_x)):
                input_x = np.repeat(batch_x[j][np.newaxis, ...], batch_size, axis=0)
                preds = model.predict(input_x, batch_size=batch_size)
                preds_y.append( vote(preds, batch_y.shape[-1]) )

        true_y = np.concatenate(true_y)
        preds_y = np.stack(preds_y)
        return evaluate_preds(true_y, preds_y)
        
    elif eval_type == 'voting':
        for i in range(len(generator)):
            batch_x, batch_y = next(iterator)
            true_y.append(batch_y)
   
            for j in range(len(batch_x) // group_size):
                input_x = np.tile(batch_x[j*group_size:(j+1)*group_size], (batch_size // group_size, 1, 1, 1))
                preds = model.predict(input_x, batch_size=batch_size)

                for k in range(group_size):
                    preds_y.append( vote(preds[k::group_size], batch_y.shape[-1]) )

        true_y = np.concatenate(true_y)
        preds_y = np.stack(preds_y)
        return evaluate_preds(true_y, preds_y)
    
    else: # 'batch' - default
        for i in range(len(generator)):
            batch_x, batch_y = next(iterator)
            true_y.append(batch_y)

            preds = model.predict(batch_x, batch_size=batch_size)
            preds_y.append(preds)

        true_y = np.concatenate(true_y)
        preds_y = np.concatenate(preds_y)
        return evaluate_preds(true_y, preds_y)
