# Imports
import argparse
import os

from keras.models import Sequential, load_model
from keras.layers import SimpleRNN, Embedding, Dropout, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, adam

from data_generator import DataGenerator
from callbacks import PlotCallback
from testing import test_model

try:
    from tensorflow.test import gpu_device_name
    
    if gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
except:
    pass

# Parameters
max_word_length = 20
hidden_size = 200
n_symbols = 31

def run(out_path, data_path, n_epochs, batch_size, model_path=None):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    train_path = data_path + '/train.txt'
    test_path = data_path + '/test.txt'
    
    # Input data
    train_generator = DataGenerator(train_path, batch_size, max_word_length)
    test_generator = DataGenerator(test_path, batch_size, max_word_length)
    
    if model_path is None:
        # Define model
        model = Sequential()
        model.add(Embedding(n_symbols, hidden_size, input_length=max_word_length))
        model.add(SimpleRNN(hidden_size, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(hidden_size, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(hidden_size))
        model.add(Dropout(0.2))
        model.add(Dense(3, activation='softmax'))
        
        optim_sgd = SGD(lr=0.001, momentum=0.9)
        optim_adam = adam(lr=0.0001)
        model.compile(loss='categorical_crossentropy', optimizer=optim_adam, metrics=['categorical_accuracy'])
    else:
        model = load_model(model_path)
    
    checkpoint_saver = ModelCheckpoint(filepath=out_path + '/model-{epoch:02d}.hdf5', verbose=True)
    
    plot_callback = PlotCallback()
    
    # Train model
    model.fit_generator(train_generator, train_generator.get_n_steps_in_epoch(), n_epochs, 
                        validation_data=test_generator,
                        validation_steps=test_generator.get_n_steps_in_epoch(), callbacks=[checkpoint_saver, plot_callback])
    
    best_model_idx = plot_callback.lowest_val_index
    best_model = load_model(out_path + '/model-{:02d}.hdf5'.format(best_model_idx+1))
    
    # Test model
    print('-- Best model')
    test_model(best_model, test_generator)
    print('-- Last model')
    test_model(model, test_generator)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_dir", required=False,
    	help="Directory where models will be saved", default='models')
    parser.add_argument("-d", "--data_dir", required=False,
    	help="Directory where data is loaded from", default='data')
    parser.add_argument("-n", "--n_epochs", required=False, type=int,
        help="Number of epochs to run", default=100)
    parser.add_argument("-b", "--batch_size", required=False, type=int,
        help="Batch size", default=512)
    parser.add_argument("-m", "--model_path", required=False,
    	help="Path to pretrained model (optional)")
    
    args = parser.parse_args()
    
    run(args.output_dir, args.data_dir, args.n_epochs, args.batch_size, args.model_path)
