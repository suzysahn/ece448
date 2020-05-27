"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import time

from math import log
import copy

import numpy as np

# Treat as Global
# index_count = 0
# tag_index = {}
# tag_counts = {}
# tag_totals = {}

def baseline(train, test):
    '''
    TODO: implement the baseline algorithm. This function has time out limitation of 1 minute.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
            test data (list of sentences, no tags on the words)
            E.g  [[word1,word2,...][word1,word2,...]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
#   raise Exception("You must implement me")
    predicts = []

    tag_totals = {}
    tag_counts = {}

    for sentence in train:
        for word, tag in sentence:
            try:
                tag_totals[tag] += 1
            except:
                tag_totals[tag] = 1
            if word not in tag_counts:
                tag_counts[word] = {}
            try:
                tag_counts[word][tag] += 1
            except:
                tag_counts[word][tag] = 1

    max_tag = getHighestValue(tag_totals)

    for sentence in test:
        prediction = []
        for word in sentence:
            best_tag = max_tag
            if word in tag_counts:
                best_tag = getHighestValue(tag_counts[word])
            prediction.append((word, best_tag))
        predicts.append(prediction)

    return predicts


def getHighestValue(tag_map):
    return max(tag_map.keys(), key=(lambda key: tag_map[key]))


def viterbi_p1(train, test):
    '''
    TODO: implement the simple Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
#   raise Exception("You must implement me")
    predicts = []

    tag_index = {}
    tag_counts = {}
    tag_totals = {}
    index_count = 0

    for sentence in train:
        for word, tag in sentence:
            try:
                tag_totals[tag] += 1
            except:
                tag_totals[tag] = 1
                tag_index[tag] = index_count
                index_count += 1
            if word not in tag_counts:
                tag_counts[word] = {}
            try:
                tag_counts[word][tag] += 1
            except:
                tag_counts[word][tag] = 1

    emission_smooth_param = 0.000001
    transition_smooth_param = 0.000001

    # tag count per word/ total tag count - emission probability
    for key in tag_counts.keys():
        for tag in tag_counts[key].keys():
            tag_counts[key][tag] = (emission_smooth_param + tag_counts[key][tag]) / (
                tag_totals[tag] + emission_smooth_param * len(tag_totals))

    initial_tag_probabilities = np.zeros(index_count)
    transition_matrix = np.zeros(shape=(index_count, index_count))

    for sentence in train:
        # first item 
        next_tag = sentence[0][1]
        curr_tag_idx = tag_index[next_tag]
        initial_tag_probabilities[curr_tag_idx] += 1

        # next items
        for i in range(len(sentence[:-1])):
            next_tag = sentence[i + 1][1]
            transition_matrix[curr_tag_idx][tag_index[next_tag]] += 1
            curr_tag_idx = tag_index[next_tag]

    # not needed
    init_smooth_param = 0.00005

    # count tag starts sentence / total num sentences
    for i in range(len(initial_tag_probabilities)):
        initial_tag_probabilities[i] = (initial_tag_probabilities[i]) / (len(train))

    # LaPlace smoothing: (1+transition occurances)/(tag occurences + num tags)
    for tag, count in tag_totals.items():
        prev_idx = tag_index[tag]
        for i in range(len(transition_matrix)):
            transition_matrix[prev_idx][i] = (transition_matrix[prev_idx][i] + transition_smooth_param) / (
                count + transition_smooth_param * len(tag_totals))

    tag_names = []
    for tag in tag_index.keys():
        tag_names.append(tag)

    for sentence in test:
        trellis = []

        for i in range(len(sentence)):
            temp = []
            curr_word = sentence[i]
            exist_word = curr_word in tag_counts

            probability = 0

            for tag in tag_index.keys():

                if exist_word and (tag in tag_counts[curr_word]):
                    probability = tag_counts[curr_word][tag]
                else:
                    probability = emission_smooth_param / (
                        tag_totals[tag] + emission_smooth_param * len(tag_totals))

                if i == 0:  # first word in sentence
                    tuple = (initial_tag_probabilities[tag_index[tag]] * probability, tag)
                    temp.append(tuple)

                else:       # next  word and beyond
                    prev_idx = tag_index[tag]
                    prev_prob = trellis[i - 1][prev_idx]
                    log3 = np.log(probability)

                    for j in range(len(tag_index)):

                        log2 = np.log(transition_matrix[prev_idx][j])
                        probability = prev_prob[0] + log2 + log3

                        tuple = (probability, tag_names[prev_idx])
                        if prev_idx == 0:
                            temp.append(tuple)

                        elif (temp[j][0] < probability):
                            temp[j] = tuple

            trellis.append(temp)

        if len(trellis) == 0:
            predicts.append([])
            continue

        tuple_list = trellis[len(trellis) - 1]
        predicted_sentence = []

        max_tag_idx = tuple_list.index((max(tuple_list, key=lambda pair: pair[0])))
        predicted_sentence.append(tag_names[max_tag_idx])

        prev_tag = max(tuple_list, key=lambda pair: pair[0])

        for i in range(len(trellis)-1, 0, -1):
            prev_tag = trellis[i - 1][tag_index[prev_tag[1]]]
            predicted_sentence.insert(0, prev_tag[1])

        max_start_tag = max(trellis[0], key=lambda pair: pair[0])[1]
        predicted_sentence[0] = max_start_tag

        predicts.append(list(zip(sentence, predicted_sentence)))

    return predicts


def viterbi_p2(train, test):
    '''
    TODO: implement the optimized Viterbi algorithm. This function has time out limitation for 3 mins.
    input:  training data (list of sentences, with tags on the words)
            E.g. [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words)
            E.g [[word1,word2...]]
    output: list of sentences with tags on the words
            E.g. [[(word1, tag1), (word2, tag2)...], [(word1, tag1), (word2, tag2)...]...]
    '''
    predicts = viterbi_p1(train, test)

    # raise Exception("You must implement me")

    return predicts
