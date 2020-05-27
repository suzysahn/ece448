"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import time

import math
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
    start = time.time()

    predicts = []
    # global index_count 
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

    max_tag = getHighestValue(tag_totals)

    print("Train = %s sec" % (time.time() - start))
    start = time.time()

    min_count = 0
    max_count = len(test)

    for sentence in test:

        min_count += 1
        print("{} / {}".format(min_count, max_count), end = '\r')

        sentence_prediction = []
        for word in sentence:
            best_tag = max_tag
            if word in tag_counts:
                best_tag = getHighestValue(tag_counts[word])
            sentence_prediction.append((word, best_tag))
        predicts.append(sentence_prediction)

    print("Testing = %s sec" % (time.time() - start))
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
#   return baseline(train, test)

    ##### DEBUG #####
    start = time.time()

    predicts = []
    #global index_count 
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

    emission_smooth_param = 0.001
    transition_smooth_param = 0.001

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
    init_smooth_param = 0.5

    # count tag starts sentence / total num sentences
    for i in range(len(initial_tag_probabilities)):
        initial_tag_probabilities[i] = (initial_tag_probabilities[i]) / (len(train))


    # LaPlace smoothing: (1+transition occurances)/(tag occurences + num tags)
    for tag, count in tag_totals.items():
        prev_idx = tag_index[tag]
        for i in range(len(transition_matrix)):
            transition_matrix[prev_idx][i] = (transition_matrix[prev_idx][i] + transition_smooth_param) / (
                count + transition_smooth_param * len(tag_totals))

    print("Train = %s sec" % (time.time() - start))
    start = time.time()

    tag_names = []
    for tag in tag_index.keys():
        tag_names.append(tag)

    progress = 0
    for sentence in test:
        progress += 1
        print("{}, {}".format(progress, time.time() - start), end = '\r')       

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

                else:       # next word and beyond
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


    print("Testing = %s sec" % (time.time() - start))
    start = time.time()

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

    predicts = []
    #raise Exception("You must implement me")
    start = time.time()
    
    #
    # https://github.com/hsuehhanhu/ECE_448_AI/blob/master/mp5-code/materials/viterbi.py
    #
    
    # 0. build 'TAG' name list     
    states = []
    no_of_tags = 0
    # 1. build Emission probability table
    emitCount = 0
    emitProb = {}
    # 2. build Initial probability dictionary
    initCount = 0
    initProb = {}
    # 3. build Transtiopn probability dictionary
    tranCount = 0
    tranProb = {}
    # 4. build and count vocabulary dictionary
    vocab_size = 0
    vocab_dict = {}
    # builing above list and dictionaries
    for sentence in train:
        prev_tag = None
        for word, tag in sentence:
            # add tag into the tag_list and probability dictionary
            if tag not in states:
                states.append(tag)
                no_of_tags += 1
                initProb[tag] = 0
                emitProb[tag] = {}            
            # counting vocabulary size
            try:
                vocab_dict[word] += 1 
            except:
                vocab_dict[word] = 1
                vocab_size += 1
            # add word into emission probability
            try:
                emitProb[tag][word] += 1
            except:
                emitProb[tag][word] = 1
            emitCount += 1
            # add into initial probability 
            if prev_tag is None:
                initProb[tag] += 1
                initCount += 1
            else:
                # build transtiopn probability
                if prev_tag not in tranProb:
                    tranProb[prev_tag] = {}
                try:
                    tranProb[prev_tag][tag] += 1
                except:
                    tranProb[prev_tag][tag] = 1
                tranCount += 1
            prev_tag = tag

        ### Needed in Part 2 ###
        words_occuring_once = 0
        for key in vocab_dict:
            if vocab_dict[key] == 1:
                words_occuring_once += 1

        woot_c = {}
        for st in states:
            count = 0
            for t in emitProb[st]:
                if emitProb[st][t] == 1:
                    count += 1
            woot_c[st] = count

    # Emission probability for P(word|tag), k = smoothing_parameter
    def getEmitProb(tag, word, k = 0.00005):
#        return (emitProb[tag].get(word, 0) + k) / (initProb[tag] + k * (vocab_size + 1))  
        return (emitProb[tag].get(word, 0) + k) / (initProb[tag] + k * (vocab_size + 1))  
    # Transtiopn probability for P(tag_curr|tag_prev), k = smoothing_parameter
    def getTranProb(prev, curr, k = 0.00005):
#        return (tranProb[prev].get(curr, 0) + k) / (initProb[prev] + k * no_of_tags)
        return (tranProb[prev].get(curr, 0) + k) / (initProb[prev] + k * emitCount)

    # Initial probability for P(tag_i|starting_position), k = smoothing_parameter
    def getInitProb(tag, k = 0.00005):
#        return (initProb.get(tag, 0) + k) / (initCount + k * no_of_tags)
        return (initProb.get(tag, 0) + k) / (initCount + k * emitCount)

 # ---------------PART TWO----------------------------

    def getEmitProb2(tag, word, k = 0.00005):
        return (emitProb[tag].get(word, 0) + k * getHapaxProb(tag)) / (initProb[tag] + k * (vocab_size + 1))  
        
    # Transtiopn probability for P(tag_curr|tag_prev), k = smoothing_parameter
    def getTranProb2(prev, curr, k = 0.00005):
        return (tranProb[prev].get(curr, 0) + k) / (initProb[prev] + (k * (len(tranProb[prev]) + 1)))

    # Initial probability for P(tag_i|starting_position), k = smoothing_parameter
    def getInitProb2(tag, k = 0.00005):
        return (initProb.get(tag, 0) + k) / (initCount + k * (len(initProb) + 1))

    def getHapaxProb(tag):
        return (woot_c[tag]/words_occuring_once)

 
    ### Needed in Part 1 & 2 ###
    start_p = {}
    for st in states:        
        start_p[st] = getInitProb2(st)

    emit_p = {}
    for st in states:
        emit_p[st] = {}
        for t in emitProb[st]:
            emit_p[st][t] = getEmitProb2(st, t)
        emit_p[st][''] = getEmitProb2(st, '')
            
    trans_p = {}
    for s1 in states:
        trans_p[s1] = {}
        for s2 in states:
            trans_p[s1][s2] = getTranProb2(s1, s2)

    ### DEBUG MESSAGE ###
    print("Train = %s sec" % (time.time() - start))

    progress = 0
    for obs in test:
        ### DEBUG MESSAGE ###
        progress += 1
        print("{}, {}".format(progress, time.time() - start), end = '\r')        
        
        V = [{}]
        for st in states:
            word = obs[0] if obs[0] in emit_p[st] else ''
            V[0][st] = {"prob": np.log(start_p[st] * emit_p[st][word]), "prev": None}

        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t-1][states[0]]["prob"] + np.log(trans_p[states[0]][st])
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t-1][prev_st]["prob"] + np.log(trans_p[prev_st][st])
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                        
                word = obs[t] if obs[t] in emit_p[st] else ''
                max_prob = max_tr_prob + np.log(emit_p[st][word])
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
                        
        opt = []
        max_prob = 0.0
        previous = None
        best_st = []
        # Get most probable state and its backtrack
        for st, data in V[-1].items():
            if data["prob"] > max_prob:
                max_prob = data["prob"]
                best_st = st
        opt.append(best_st)
        previous = best_st
        
        # Follow the backtrack till the first observation
        for t in range(len(V) - 2, -1, -1):
            opt.insert(0, V[t + 1][previous]["prev"])
            previous = V[t + 1][previous]["prev"]

        # print ('The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob)
        opt = list(zip(obs, opt))
        predicts.append(opt)
    
    print("Testing = %s sec" % (time.time() - start))
    start = time.time()

    return predicts
