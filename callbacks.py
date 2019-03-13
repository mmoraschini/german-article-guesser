import copy

import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import Callback

import os

class PlotCallback(Callback):
    def __init__(self, step=1, fallback_save_dir=None):
        self.step = step
        
        if fallback_save_dir is None:
            self.fallback_save_dir = os.path.expanduser('~')
        else:
            self.fallback_save_dir = fallback_save_dir
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        
        self.logs = []
        
        self.lowest_val_loss = np.inf
        self.lowest_val_index = -1

    def on_epoch_end(self, epoch, logs={}):
        
        if self.i % self.step == 0:
            # Logs are written to the same memory address at each iteration.
            # If you just append them, you are appending the same reference
            # over and over again
            self.logs.append(copy.copy(logs))
            self.x.append(self.i)
            
            if logs['val_loss'] < self.lowest_val_loss:
                self.lowest_val_loss = logs['val_loss']
                self.lowest_val_index = self.i
        
        self.i += 1
    
    def on_train_end(self, logs={}):
        
        # Check if there is an x-server running
        video = True
        try:
            os.environ['DISPLAY']
        except KeyError:
            video = False
        
        metrics = np.array(list(self.logs[0].keys()))
        val_metrics_bool = [x.startswith('val') for x in metrics]
        
        val_metrics = np.sort(metrics[val_metrics_bool])
        not_val_metrics = np.sort(metrics[np.logical_not(val_metrics_bool)])
        
        for i in range(len(val_metrics)):
            vm = val_metrics[i]
            nvm = not_val_metrics[i]
            
            vm_data = [l[vm] for l in self.logs]
            nvm_data = [l[nvm] for l in self.logs]
            
            fig = plt.figure()
            plt.plot(self.x, vm_data, label=vm)
            plt.plot(self.x, nvm_data, label=nvm)
                
            plt.legend()

            if video == True:
                plt.show()
            else:
                print('Saving plots to ' + self.fallback_save_dir)
                fig.savefig(self.fallback_save_dir + '/' + nvm + '.png')
        
        print('Lowest validation loss = {:.4f} at index {}'.format(self.lowest_val_loss, self.lowest_val_index))
