# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time

# Create data for number of docs in train set containing word w
def createDocumentFrequency(train_set):
    # Add labeled words into wordList and count numbers
    docuDict = {}
    for i in range(len(train_set)):
        email = set(train_set[i])
        for word in email:
            docuDict[word] = docuDict.get(word, 0) + 1
    return docuDict


# Create data for number of times word w appears in doc. A
def createEmailFrequency(email):
    # Add labeled words into wordList and count numbers
    wordDict = {}
    for word in email:
        wordDict[word] = wordDict[word] + 1 if word in wordDict else 1
    return wordDict


def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """

    # TODO: Write your code here
    # start_time = time.time()

    # Create data for number of docs in train set containing word w
    docFrequency = createDocumentFrequency(train_set)

    # Unigram Model
    dev_labels = []
    tf_r_top = (len(train_set) + 1)

    for email in dev_set:
        emailTotalWords = len(email)
        emailWordFrequency = createEmailFrequency(email)

        high_idf = 0
        high_word = ""

        for word in email:
            tf_left = (emailWordFrequency[word] if word in emailWordFrequency else 0) / emailTotalWords
            tf_r_btm = 1 + (docFrequency[word] if word in docFrequency else 0)
            tf_right = math.log(tf_r_top / tf_r_btm)

            tf_idf = tf_left * tf_right
            if tf_idf > high_idf:
                high_idf = tf_idf
                high_word = word

        dev_labels.append(high_word)

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    # elapsed_time = time.time() - start_time
    # print("Elasped: {0} sec".format(elapsed_time))
    return dev_labels