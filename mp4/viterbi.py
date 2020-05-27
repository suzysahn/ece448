"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
from collections import Counter

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
    # Counters to keep track of number of tags of seen words
    tag_with_words = {}
    # Counter to keep track of number of tags of all types appeared in sentence
    tags = Counter()

    # Train Data
    for sentence in train:
        for pair in sentence:
            word = pair[0]
            tag = pair[1]

            # If new word, give it a dictionary to store types of tags to count
            if word not in tag_with_words:
                tag_with_words[word] = Counter()

            # Record word_tag's type count in tags
            tags[tag] += 1

            # If the tag type is new for word, set to 1
            if tag not in tag_with_words[word]:
                tag_with_words[word][tag] = 1
            else:
                tag_with_words[word][tag] += 1

    # List of sentences: (word,tag) pairs to be returned
    predicts = []

    # Develop Data
    for sentence in test:
        sentence_predicted = []
        for word in sentence:
            # Create the most common pair with most common tag of all tags
            if word not in tag_with_words:
                most_common_pair = (word, tags.most_common(1)[0][0])
            # Create the most common pair with most common tag of specific word
            else:
                most_common_pair = (word, tag_with_words[word].most_common(1)[0][0])
            sentence_predicted.append(most_common_pair)
        predicts.append(sentence_predicted)

    return predicts

import math

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
    # Assign smoothing factor (as suggested on piazza)
    k = 1e-5

    # Dictionary to keep count of occurrence
    words = Counter()  # (word, count)
    tags = Counter()  # (tag, count)
    word_to_tag = Counter()  # ((word, tag), count)
    prev_and_cur = Counter()  # ((cur, prev), count)
    init_tag = Counter()  # (first_word_of_sentence, count)

    # Train Data
    for sentence in train:
        # increment first word of sentence's tag
        init_tag[sentence[0][1]] += 1
        max_idx = len(sentence) - 1
        pair_idx = 0
        for pair in sentence:
            cur_word = pair[0]
            cur_tag = pair[1]

            # Increment word, tag, (word, tag) count
            words[cur_word] += 1
            tags[cur_tag] += 1
            word_to_tag[(cur_word, cur_tag)] += 1

            # Increment (cur,next) count
            if pair_idx + 1 < max_idx:
                prev_and_cur[(cur_tag, sentence[pair_idx + 1][1])] += 1
            pair_idx += 1

    # Dictionaries to keep probabilities of each type:
    trans_prob = {}  # Transmission Probability
    init_prob = {}  # Initial Probability

    # Calculate the Transmission and Initial Probabilities
    # Formulas given on piazza
    for cur_tag in tags:
        for prev_tag in tags:
            n = prev_and_cur[(prev_tag, cur_tag)] + k
            d = tags[prev_tag] + k * len(tags)
            trans_prob[(cur_tag, prev_tag)] = math.log(n / d)

        n = init_tag[cur_tag] + k
        d = sum(init_tag.values()) + k * len(tags)
        init_prob[cur_tag] = math.log(n / d)

    predicts = []
    # Develop Data:
    for sentence in test:
        # algorithm from page 11 of https://web.stanford.edu/~jurafsky/slp3/8.pdf
        viterbi = {}
        backpointer = {}
        first_word = sentence[0]

        # Initialization
        for cur_tag in tags:
            n = word_to_tag[(first_word, cur_tag)] + k
            d = tags[cur_tag] + k * len(words)
            emission_prob = math.log(n / d)

            viterbi[(cur_tag, 0)] = init_prob[cur_tag] + emission_prob
            backpointer[(cur_tag, 0)] = 0

        # Recursion
        best_path_max = {}
        max_time_step = range(len(sentence))
        max_len = len(sentence) - 1
        for t in max_time_step:
            if t is 0:
                continue
            for cur_tag in tags:
                # get max probability of previous time step
                prob = {}
                for prev_tag in tags:
                    n = word_to_tag[(sentence[t], cur_tag)] + k
                    d = tags[cur_tag] + k * len(words)
                    emission_prob = math.log(n / d)
                    prob[prev_tag] = trans_prob[(cur_tag, prev_tag)] + emission_prob + viterbi[(prev_tag, t - 1)]

                # get max prob from previous time step
                viterbi[(cur_tag, t)] = max(prob.values())

                # get key tag of the maximum into back pointer
                # https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
                backpointer[(cur_tag, t)] = (max(prob, key=prob.get), t - 1)

        # Termination step:
        # get the list of best probs for each tag at max_len
        for cur_tag in tags:
            best_path_max[(cur_tag, max_len)] = viterbi[(cur_tag, max_len)]

        best_path = []
        # Get index of the best probability
        bestpathpointer = max(best_path_max, key=best_path_max.get)
        # follows backpointer[] to states back in time
        while bestpathpointer != 0:
            best_path.insert(0, bestpathpointer[0])
            bestpathpointer = backpointer[bestpathpointer]

        predicted_sentence = []
        for idx in max_time_step:
            predicted_sentence.append((sentence[idx], best_path[idx]))

        predicts.append(predicted_sentence)
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
    # Assign smoothing factor (as suggested on piazza)
    k = 1e-7

    # Dictionary to keep count of occurrence
    words = Counter()               # (word, count)
    tags = Counter()                # (tag, count)
    word_to_tag = Counter()         # ((word, tag), count)
    prev_and_cur = Counter()        # ((cur, prev), count)
    init_tag = Counter()            # (first_word_of_sentence, count)
    hapax_word_tags = Counter()     # (hapax-word-tag, count)

    # Train Data
    for sentence in train:
        # Increment first word's tag count for initial probability
        init_tag[sentence[0][1]] += 1

        # Get index max and initialize idx calculator
        max_idx = len(sentence) - 1
        pair_idx = 0

        # Populate all the counters with data
        for pair in sentence:
            cur_word = pair[0]
            cur_tag = pair[1]

            # Increment word, tag, (word, tag) count
            words[cur_word] += 1
            tags[cur_tag] += 1
            word_to_tag[(cur_word, cur_tag)] += 1

            # Increment (cur,next) count
            if pair_idx is max_idx:
                continue
            else:
                pair_idx += 1
                prev_and_cur[(cur_tag, sentence[pair_idx][1])] += 1

    # Count hapax words and its tags
    hapax_words = []
    # Create list of hapax words to calculate probabilities
    for word in words:
        if words[word] is 1:
            hapax_words.insert(0, word)

    # Find and incrememnt those tags
    for tag in tags:
        for hap_word in hapax_words:
            hapax_word_tags[tag] += word_to_tag[(hap_word, tag)]

    # Dictionaries to keep probabilities of each type:
    trans_prob = {}     # Transmission Probability
    init_prob = {}      # Initial Probability
    hapax_prob = {}     # Hapax word tag probabilities

    # Calculate the Transmission, Initial, and Hapax word-tag Probabilities
    # Formulas given on piazza
    for cur_tag in tags:
        for prev_tag in tags:
            n = prev_and_cur[(prev_tag, cur_tag)] + k
            d = tags[prev_tag] + k * len(tags)
            trans_prob[(cur_tag, prev_tag)] = math.log(n / d)

        n = init_tag[cur_tag] + k
        d = sum(init_tag.values()) + k * len(tags)
        init_prob[cur_tag] = math.log(n / d)

        n = hapax_word_tags[cur_tag] + k
        d = len(hapax_words) + k * len(tags)
        hapax_prob[cur_tag] = n/d

    predicts = []
    # Develop Data:
    for sentence in test:
        first_word = sentence[0]

        # algorithm from page 11 of https://web.stanford.edu/~jurafsky/slp3/8.pdf
        viterbi = {}
        backpointer = {}

        # Initialization
        for cur_tag in tags:
            # Get hapax distribution factor
            hapax_k = k * hapax_prob[cur_tag]

            # Calculate emission with new smoothing
            n = word_to_tag[(first_word, cur_tag)] + hapax_k
            d = tags[cur_tag] + hapax_k * len(words)
            emission_prob = math.log(n / d)

            # Populate dictionaries with probability and prev tag pointer for each tag
            viterbi[(cur_tag, 0)] = init_prob[cur_tag] + emission_prob
            backpointer[(cur_tag, 0)] = 0

        # Recursion
        best_path_max = {}
        max_len = len(sentence) - 1
        for t in range(max_len + 1):
            if t is 0:
                continue
            for cur_tag in tags:
                # Get hapax distribution factor
                hapax_k = k * hapax_prob[cur_tag]

                # get max probability of previous time step
                prev_maxProb = {}
                for prev_tag in tags:
                    n = word_to_tag[(sentence[t], cur_tag)] + hapax_k
                    d = tags[cur_tag] + hapax_k * len(words)
                    emission_prob = math.log(n / d)

                    prev_maxProb[prev_tag] = trans_prob[(cur_tag, prev_tag)] + emission_prob + viterbi[(prev_tag, t - 1)]

                # get max prob from previous time step
                viterbi[(cur_tag, t)] = max(prev_maxProb.values())

                # get key tag of the maximum into back pointer
                # https://www.geeksforgeeks.org/python-get-key-with-maximum-value-in-dictionary/
                backpointer[(cur_tag, t)] = (max(prev_maxProb, key=prev_maxProb.get), t - 1)

        # Termination step:
        # get the list of best probs for each tag at max_len
        for cur_tag in tags:
            best_path_max[(cur_tag, max_len)] = viterbi[(cur_tag, max_len)]

        best_path = []
        # Get index of the best probability
        bestpathpointer = max(best_path_max, key=best_path_max.get)
        # follows backpointer[] to states back in time
        while bestpathpointer is not 0:
            best_path.insert(0, bestpathpointer[0])
            bestpathpointer = backpointer[bestpathpointer]

        predicted_sentence = []
        pair_idx = 0
        for pair in sentence:
            predicted_sentence.append((sentence[pair_idx], best_path[pair_idx]))
            pair_idx += 1

        predicts.append(predicted_sentence)

    return predicts