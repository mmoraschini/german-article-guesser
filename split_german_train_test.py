import argparse
import unicodedata
import re
import os

import numpy as np

def is_valid(word, dictionary):
    ''' Checks that "word" is not a coumpoud word, whose ending word
    will be added to the data'''
    for w in dictionary:
        if word != w and word.lower().endswith(w.lower()):
            return False
    return True

def split_train_test_data(file_path, output_dir, perc_test):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    last_word = ''
    
    word_list = []
    gender_dict = {}
    with open(file_path, 'r') as f:
        
        print('Working on it...')
        
        line = f.readline()
        
        while line is not None and line != '':
            
            # Thanks to https://gist.github.com/j4mie/557354 for this line
            first_char = unicodedata.normalize('NFKD', line[0]).encode('ASCII', 'ignore').decode('utf-8')
            
            if re.match('[A-Z]', first_char):
                tok = line.strip().split()
                if tok[-1] == 'noun':
                    gender = tok[1]
                    
                    # Skipping plurals
                    if gender in ['{m}', '{f}', '{n}']:
                        
                        word = tok[0]
                        
                        # Skip words with hypens, apostrophes, dots, etc.
                        if re.match('^[a-zäöüß]+$', word.lower()) and word != last_word:
                            word_list.append(word)
                            gender_dict[word] = gender[1]
                            
                            last_word = word
            
            line = f.readline()
    
    word_list.sort(key=len)
    
    with open(output_dir + '/train.txt', 'w') as tr:
        with open(output_dir + '/test.txt', 'w') as te:
            for i in range(len(word_list)):
                print('{}/ {}'.format(i, len(word_list)))
                w = word_list[i]
                if i > 0 and not is_valid(w, word_list[:i]):
                    continue
                
                r = np.random.rand()
                if r < perc_test / 100:
                    te.write(w + '\t' + gender_dict[w] + '\n')
                else:
                    tr.write(w + '\t' + gender_dict[w] + '\n')
    
    print('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--file_in", required=True,
    	help="Input file")
    parser.add_argument("-o", "--output_dir", required=True,
    	help="Directory where output files will be saved")
    parser.add_argument("-p", "--perc_test", required=False, type=int,
        help="Percentage of data for testing", default=10)
    
    args = parser.parse_args()
    
    split_train_test_data(args.file_in, args.output_dir, args.perc_test)