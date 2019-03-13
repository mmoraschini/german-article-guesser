import numpy as np
import re

def str_to_int(string):
    
    num_str = np.empty(len(string), dtype=np.int)
    for i in range(len(string)):
        c = string[i]
        if re.match('[a-z]', c):
            num_str[i] = ord(c) - 97
        elif c == 'ä':
            num_str[i] = 26
        elif c == 'ö':
            num_str[i] = 27
        elif c == 'ü':
            num_str[i] = 28
        elif c == '-':
            num_str[i] = 29
        elif c == 'ß':
            num_str[i] = 30
        else:
            raise(RuntimeError('Character not valid: ' + c))
        
    return num_str

class DataGenerator(object):
    def __init__(self, file_path, batch_size, max_length):
        
        self.batch_size = batch_size
        self._max_length = max_length
        
        i = 0
        with open(file_path, 'r') as f:
            for line in f:
                tok = line.strip().split()
                if len(tok[0]) <= max_length:
                    i += 1
        
        self._size = i
        
        self._file_handle = open(file_path, 'r')
    
    def get_data_size(self):
        return self._size
    
    def get_n_steps_in_epoch(self):
        
        if self._size % self.batch_size == 0:
            n_steps = self._size // self.batch_size
        else:
            n_steps = self._size // self.batch_size + 1
        
        return n_steps
    
    def reload(self):
        self._file_handle.seek(0)
#        self._file_handle.close()
#        self._file_handle = open(self._file_handle.name, 'r')
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        X = np.empty((self.batch_size, self._max_length), dtype=np.int)
        Y = np.empty((self.batch_size, 3), dtype=np.int)
        
        stop = False
        i = 0
        while stop == False:
            line = self._file_handle.readline()
            
            if line is None or line == '':
                self._file_handle.close()
                self._file_handle = open(self._file_handle.name, 'r')
                line = self._file_handle.readline()
            
            try:
                x, y = line.strip().split()
            except:
                raise(RuntimeError('This line didn\'t work: "' + line + '"'))
            
            if len(x) > self._max_length:
                continue
            
            x = x.lower().ljust(self._max_length, '-')
            
            X[i, :] = str_to_int(x)
            
            if y == 'm':
                Y[i,:] = [1,0,0]
            elif y == 'f':
                Y[i,:] = [0,1,0]
            elif y == 'n':
                Y[i,:] = [0,0,1]
            else:
                raise(RuntimeError('Gender not possible: ' + y))
            
            i += 1
            
            if i == self.batch_size:
                stop = True
        
        return X, Y