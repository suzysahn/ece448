# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""
import time
import math
import queue
import numpy as np

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters
    b = 0
    W = [0]*3072

    for i in range(max_iter):
        for vector, label in zip(train_set, train_labels):
            model = 1 if (np.dot(W, vector) + b) > 0 else 0
            W += learning_rate * (label - model) * vector 
            b += learning_rate * (label - model)

    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    start = time.time()    
    
    W, b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    
    dev_labels = []
    for dev_vector in dev_set:
        model = 1 if (np.dot(W, dev_vector) + b) > 0 else 0
        dev_labels.append(model)
    
    print("classifyPerceptron = %s sec" % (time.time() - start))
    return dev_labels  
    # return []


def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    return 1 / (1 + math.exp(-x))

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters 
    b = 0
    W = [0]*3072
    wsum = 0

    for i in range(max_iter):
        for vector, label in zip(train_set, train_labels):
            # model = 1 if (sigmoid(np.dot(W, vector) + b)) > 0 else 0
            model = sigmoid(np.dot(W, vector) + b)
            # model is going to be 0.5 or greater = 1 (classified), less than .5 then = 0
            wsum += (model - label) * vector

        W -= learning_rate * wsum / len(train_labels)
        b -= learning_rate * (label - model)
    
    return trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    # return W, b

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    start = time.time()    
    
    W, b = trainLR(train_set, train_labels, learning_rate, max_iter)
    
    dev_labels = []
    for dev_vector in dev_set:
        model = 1 if (np.dot(W, dev_vector) + b) > 0 else 0
        dev_labels.append(model)
    
    print("classifyLR = %s sec" % (time.time() - start))      
    return dev_labels
    
    # return classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter)
    # return []

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    # k == mmax_iter
    start = time.time()   
    
    results = []
    for dev_id, dev_elem in enumerate(dev_set):
        heap = queue.PriorityQueue()
        for vector_id, vector in enumerate(train_set):
            diff = math.sqrt(np.sum(pow(vector-dev_elem,2)))
            tuple = (diff, train_labels[vector_id])
            heap.put((diff,train_labels[vector_id]))
            while heap.qsize() > k:
                heap.get()
        sum = 0
        while heap.qsize():
            sum += heap.get()[1]
        label = round(sum/k)
        results.append(label)
    
    print("classifyEC = %s sec" % (time.time() - start))   
    return results;
    # return []
