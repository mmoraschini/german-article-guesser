# Summary
With the code in this project you can train and test a Recurrent Neural Network to try and guess the correct gender of German nouns.

# Setup
To run it first you have to download translation data from dict.cc at https://www1.dict.cc/translation_file_request.php?l=e

Then you have to run _split_german_train_test.py_ with the path to the file you downloaded, the output drectory and (optional) the percentage of data for the testing set (int).
This script filters out all words that are either not nouns, plural or contain characters that are not a-z, ä, ö, ü and ß. These include apostrophes, hypens, dots, etc. Moreover it removes all words that and with another word, i.e. compound words. This must be done because 1) the gender of a compound word is only given by the last word and 2) we don't want to have words with the same ending splitted into training and testing set. This will take a while to complete.

# Running
After you generated the train and test sets you can run the script _german_article_guesser.py_ that will train a RNN on words in the training file and will test them on words in the testing file.

You can adjust the parameter _max_word_length_ (default: 20) inside _german_article_guesser.py_: words that are shorter than this number will be padded on the right with '-' to reach the chosen length, words that are longer will be skipped.

With default parameters I reached an accuracy of ~85% on the testing set.

![Train and test loss](images/loss.png?raw=true "Train and test loss")
![Train and test accuracy](images/accuracy.png?raw=true "Train and test accuracy")

# Testing
You can test the trained network on a file or on a list of words with _test_words.py_
