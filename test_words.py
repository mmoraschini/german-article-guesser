import argparse

import numpy as np

from keras.models import load_model

from data_generator import str_to_int

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True,
    	help="Path to the model to test")
    parser.add_argument("-f", "--file", required=False,
    	help="Input file with words to test. Each line must contain a word and the gender separated by <tab>", default=None)
    parser.add_argument('-w','--words', nargs='+', required=False, 
        help='Words to test', default=None)
    
    args = parser.parse_args()
    
    if args.file is not None and args.words is not None:
        raise(RuntimeError('You must either pass in a sequence of words or a file, not both'))
    if args.file is None and args.words is None:
        raise(RuntimeError('You must pass in a sequence of words or a file, no input specified'))
    
    model = load_model(args.model_path)
    
    shape = model.input.get_shape().as_list()[1]
    
    if args.words is not None:
        gender_dict = {0:'masculine', 1:'feminine', 2:'neuter'}
        
        l = len(args.words)
        test_data = np.empty((l, shape), dtype=np.int)
        for i in range(l):
            word = args.words[i]
            if len(word) > shape:
                raise(RuntimeError('This model accepts words at most {} big'.format(shape)))
            
            word = word.lower().ljust(shape, '-')
            test_data[i,:] = str_to_int(word)
        
        y = model.predict(test_data)
        for i in range(len(args.words)):
            word = args.words[i]
            pred_gender = gender_dict[np.argmax(y[i])]
            print('I guess that {} is {}'.format(word, pred_gender))
    else:
        gender_dict = {'m':0, 'f':1, 'n':2}
        
        total = 0
        with open(args.file, 'r') as f:
            for l in f:
                if l is None or l == '':
                    break
                
                tok = l.strip().split()
                if len(tok[0]) > shape:
                    continue
                total += 1
        
        test_data = np.empty((total, shape), dtype=np.int)
        y_true = np.empty(total, dtype=np.int)
        
        i = 0
        with open(args.file, 'r') as f:
            for l in f:
                if l is None or l == '':
                    break
                
                tok = l.strip().split()
                if len(tok[0]) > shape:
                    continue
                
                word = tok[0].lower().ljust(shape, '-')
                gender = gender_dict[tok[1]]
                
                test_data[i,:] = str_to_int(word)
                y_true[i] = gender
                
                i += 1
        
        y_pred = np.argmax(model.predict(test_data), axis=1)
        
        correct = np.sum(y_pred == y_true)
    
        print('Correct: {}, total: {}'.format(correct, total))
        print('Percentage of correct answers: {:.2f}%'.format(correct / total * 100))