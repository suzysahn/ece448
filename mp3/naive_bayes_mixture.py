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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
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


def pairwise(iterable):
    a = iter(iterable)
    return zip(a, a)


# creates list of words with corresponding number = ('cat'in spam emails / all words in spam emails)
def createBiProbabilitiesList(train_set, train_labels, label, smoothing_param):
    # Add labeled words into wordList and count numbers
    wordDict = {}
    total_words = 0

    # Count words into wordDict
    for i in range(len(train_labels)):
        if (train_labels[i] != label):
            continue
        email = train_set[i]
        for item in [email[i:i + 2] for i in range(len(email) - 1)]:
            word = str(item)
            total_words += 1
            wordDict[word] = wordDict[word] + 1 if word in wordDict else 1
                
    # Build probabilities from word list
    probFactor = total_words + smoothing_param * (len(wordDict) + 1)
    probUnknown = math.log(smoothing_param / probFactor)

    # Change count as probabilities
    for word, count in wordDict.items():
        wordDict[word] = math.log((count + smoothing_param) / probFactor)
    return wordDict, probUnknown


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda, unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # start_time = time.time()

    # create UniGram probabilities from train set
    posProbWords, posProbUnknown = createProbabilitiesList(train_set, train_labels, 1, unigram_smoothing_parameter)    
    negProbWords, negProbUnknown = createProbabilitiesList(train_set, train_labels, 0, unigram_smoothing_parameter)

    # Unigram Model
    pro_labels = []
    neg_labels = []

    for email in dev_set:
        negProbability = 0.0
        posProbability = 0.0

        for word in email:
            posProbability += posProbWords[word] if word in posProbWords else posProbUnknown   
            negProbability += negProbWords[word] if word in negProbWords else negProbUnknown

        #posProbability += math.log(pos_prior)
        pro_labels.append(posProbability)
        neg_labels.append(negProbability)

    # create BiGram probabilities from train set
    posProbWords2, posProbUnknown2 = createBiProbabilitiesList(train_set, train_labels, 1, bigram_smoothing_parameter)    
    negProbWords2, negProbUnknown2 = createBiProbabilitiesList(train_set, train_labels, 0, bigram_smoothing_parameter)

    # BiGram Model
    pro_labels2 = []
    neg_labels2 = []

    for email in dev_set:
        negProbability = 0.0
        posProbability = 0.0

        for item in [email[i:i + 2] for i in range(len(email) - 1)]:
            word = str(item)
            posProbability += posProbWords2[word] if word in posProbWords2 else posProbUnknown2   
            negProbability += negProbWords2[word] if word in negProbWords2 else negProbUnknown2

        pro_labels2.append(posProbability)
        neg_labels2.append(negProbability)

    # Build BiGram into dev_lables
    dev_labels = []

    # Weights the models (1-lambda) multiplier for unigram and lamba multiplier for bigram
    LAMBDA = bigram_lambda
    for i in range(len(dev_set)):
        posProbability = (1 - LAMBDA) * pro_labels[i] + LAMBDA * pro_labels2[i]
        negProbability = (1 - LAMBDA) * neg_labels[i] + LAMBDA * neg_labels2[i]

        posProbability += math.log(pos_prior)
        dev_labels.append(0 if negProbability > posProbability else 1)

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    # elapsed_time = time.time() - start_time
    # print("Elasped: {0} sec".format(elapsed_time))
    return dev_labels