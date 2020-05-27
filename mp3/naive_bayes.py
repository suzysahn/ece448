# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
# for the time test
# import time

# creates list of words with corresponding number = ('cat'in spam emails / all words in spam emails)
def createProbabilitiesList(train_set, train_labels, label, smoothing_param):
    # Add labeled words into wordList and count numbers
    wordDict = {}
    total_words = 0

    # Count words into wordDict
    for i in range(len(train_labels)):
        if (train_labels[i] != label):
            continue
        for word in train_set[i]:
            total_words += 1
            wordDict[word] = wordDict[word] + 1 if word in wordDict else 1
                
    # Build probabilities from word list
    probFactor = total_words + smoothing_param * (len(wordDict) + 1)
    probUnknown = math.log(smoothing_param / probFactor)

    # Change count as probabilities
    for word, count in wordDict.items():
        wordDict[word] = math.log((count + smoothing_param) / probFactor)
    return wordDict, probUnknown

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # start_time = time.time()

    # create probabilities from train set
    posProbWords, posProbUnknown = createProbabilitiesList(train_set, train_labels, 1, smoothing_parameter)    
    negProbWords, negProbUnknown = createProbabilitiesList(train_set, train_labels, 0, smoothing_parameter)

    # Unigram Model
    dev_labels = []

    for i in range(len(dev_set)):
        negProbability = 0.0
        posProbability = 0.0

        for word in dev_set[i]:
            posProbability += posProbWords[word] if word in posProbWords else posProbUnknown   
            negProbability += negProbWords[word] if word in negProbWords else negProbUnknown

        posProbability += math.log(pos_prior)
        dev_labels.append(0 if negProbability > posProbability else 1)

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    # elapsed_time = time.time() - start_time
    # print("Elasped: {0} sec".format(elapsed_time))
    return dev_labels