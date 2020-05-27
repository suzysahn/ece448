"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import time

import math
import copy
import numpy as np

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

    # 0. build each 'TAG''s frequency     
    tagTotals = {}
    # 1. build each words count by tags
    tagCounts = {}
    # builing above list and dictionaries
    for sentence in train:
        for word, tag in sentence:
            # add tag into the tagTotals and count
            try:
                tagTotals[tag] += 1
            except:
                tagTotals[tag] = 1
            # add word into tagCounts and count
            if word not in tagCounts:
                tagCounts[word] = {}
            try:
                tagCounts[word][tag] += 1
            except:
                tagCounts[word][tag] = 1

    max_tag = max(tagTotals.keys(), key=(lambda key: tagTotals[key]))

    ### DEBUG MESSAGE ###
    print("Train = %s sec" % (time.time() - start))

    progress = 0
    for sentence in test:
        ### DEBUG MESSAGE ###
        progress += 1
        print("{}, {}".format(progress, time.time() - start), end = '\r')     

        sentence_prediction = []
        for word in sentence:
            best_tag = max_tag
            if word in tagCounts:
                best_tag = max(tagCounts[word].keys(), key=(lambda key: tagCounts[word][key]))
            sentence_prediction.append((word, best_tag))
        predicts.append(sentence_prediction)

    print("Total = %s sec" % (time.time() - start))
    return predicts


def viterbi_p1(train, test):
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

    # Emission probability for P(word|tag), k = smoothing_parameter
    def getEmitProb(tag, word, k = 0.01):
        return (emitProb[tag].get(word, 0) + k) / (initProb[tag] + k * (vocab_size + 1))  

    # Transtiopn probability for P(tag_curr|tag_prev), k = smoothing_parameter
    def getTranProb(prev, curr, k = 0.01):
        return (tranProb[prev].get(curr, 0) + k) / (initProb[prev] + k * no_of_tags)

    # Initial probability for P(tag_i|starting_position), k = smoothing_parameter
    def getInitProb(tag, k = 0.01):
        return (initProb.get(tag, 0) + k) / (initCount + k * no_of_tags)

    start_p = {}
    for st in states:        
        start_p[st] = np.log(getInitProb(st))

    emit_p = {}
    for st in states:
        emit_p[st] = {}
        for t in emitProb[st]:
            emit_p[st][t] = np.log(getEmitProb(st, t))
        emit_p[st][''] = np.log(getEmitProb(st, ''))
            
    trans_p = {}
    for s1 in states:
        trans_p[s1] = {}
        for s2 in states:
            trans_p[s1][s2] = np.log(getTranProb(s1, s2))

    ### DEBUG MESSAGE ###
    print("Train = %s sec" % (time.time() - start))

    progress = 0
    for obs in test:
        ### DEBUG MESSAGE ###
        progress += 1
        # print("{}, {}".format(progress, time.time() - start), end = '\r')        
        
        V = [{}]
        for st in states:
            word = obs[0] if obs[0] in emit_p[st] else ''
            V[0][st] = {"prob": start_p[st] + emit_p[st][word], "prev": None}

        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t-1][states[0]]["prob"] + trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t-1][prev_st]["prob"] + trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                        
                word = obs[t] if obs[t] in emit_p[st] else ''
                max_prob = max_tr_prob + emit_p[st][word]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
                        
        opt = []
        max_prob = -99999999999
        previous = None
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
    
    print("Total = %s sec" % (time.time() - start))
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
       return (emitProb[tag].get(word, 0) + k) / (initProb[tag] + k * (vocab_size + 1))  

    # Transtiopn probability for P(tag_curr|tag_prev), k = smoothing_parameter
    def getTranProb(prev, curr, k = 0.00005):
       return (tranProb[prev].get(curr, 0) + k) / (initProb[prev] + k * no_of_tags)

    # Initial probability for P(tag_i|starting_position), k = smoothing_parameter
    def getInitProb(tag, k = 0.00005):
       return (initProb.get(tag, 0) + k) / (initCount + k * no_of_tags)


 # ---------------PART TWO----------------------------

    def getEmitProb2(tag, word, k = 0.01):
        return (emitProb[tag].get(word, 0) + k * getHapaxProb(tag)) / (initProb[tag] + k * (vocab_size + 1))  
        
    # Transtiopn probability for P(tag_curr|tag_prev), k = smoothing_parameter
    def getTranProb2(prev, curr, k = 0.01):
        return (tranProb[prev].get(curr, 0) + k * getHapaxProb(prev)) / (initProb[prev] + (k * (len(tranProb[prev]) + 1)))

    # Initial probability for P(tag_i|starting_position), k = smoothing_parameter
    def getInitProb2(tag, k = 0.01):
        return (initProb.get(tag, 0) + k* getHapaxProb(tag)) / (initCount + k * (len(initProb)+1))

    def getHapaxProb(tag):
        return (woot_c[tag]/words_occuring_once)

 
    ### Needed in Part 1 & 2 ###
    start_p = {}
    for st in states:        
        start_p[st] = np.log(getInitProb2(st))

    emit_p = {}
    for st in states:
        emit_p[st] = {}
        for t in emitProb[st]:
            emit_p[st][t] = np.log(getEmitProb2(st, t))
        emit_p[st][''] = np.log(getEmitProb2(st, ''))
            
    trans_p = {}
    for s1 in states:
        trans_p[s1] = {}
        for s2 in states:
            trans_p[s1][s2] = np.log(getTranProb2(s1, s2))

    ### DEBUG MESSAGE ###
    print("Train = %s sec" % (time.time() - start))

    progress = 0
    for obs in test:
        ### DEBUG MESSAGE ###
        progress += 1
        # print("{}, {}".format(progress, time.time() - start), end = '\r')        
        
        V = [{}]
        for st in states:
            word = obs[0] if obs[0] in emit_p[st] else ''
            V[0][st] = {"prob": start_p[st] + emit_p[st][word], "prev": None}

        # Run Viterbi when t > 0
        for t in range(1, len(obs)):
            V.append({})
            for st in states:
                max_tr_prob = V[t-1][states[0]]["prob"] + trans_p[states[0]][st]
                prev_st_selected = states[0]
                for prev_st in states[1:]:
                    tr_prob = V[t-1][prev_st]["prob"] + trans_p[prev_st][st]
                    if tr_prob > max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = prev_st
                        
                word = obs[t] if obs[t] in emit_p[st] else ''
                max_prob = max_tr_prob + emit_p[st][word]
                V[t][st] = {"prob": max_prob, "prev": prev_st_selected}
                        
        opt = []
        max_prob = -99999999
        previous = None
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
