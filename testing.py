import numpy as np

def test_model(model, data_generator):
    '''
    Tests given model on specified data
    
    --- args
    model: model to test
    data_generator: a generator of data to be tested
    '''
    
    data_generator.reload()
    
    data_size = data_generator.get_data_size()
    n_steps = data_generator.get_n_steps_in_epoch()
        
    y_pred = np.empty(data_size, dtype=np.int)
    y_true = np.empty(data_size, dtype=np.int)
    
    b = data_generator.batch_size
    
    for i in range(n_steps):
        [x, y] = next(data_generator)
        
        output = model.predict(x)
        
#        l = y_pred[i*b:(i+1)*b].shape[0]
        
        start = i*b
        end = min(y_pred.shape[0], (i+1)*b)
        
        l = end - start
        
        y_pred[start:end] = np.argmax(output[:l], axis=1)
        y_true[start:end] = np.argmax(y[:l], axis=1)
        
    correct = np.sum(y_pred == y_true)
    total = y_true.shape[0]
    
    print('Correct: {}, total: {}'.format(correct, total))
    
    print('Percentage of correct answers: {:.2f}%'.format(correct / total * 100))
    